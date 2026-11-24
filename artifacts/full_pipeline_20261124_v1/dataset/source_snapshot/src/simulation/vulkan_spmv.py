"""Optional fp64 Vulkan sparse products for the existing CPU CG solver.

This is a hybrid experiment: assembly, preconditioning, CG vectors/reductions,
nonlinear flow and scalar transport stay on the CPU. Every product includes a
host write, queue submission, fence wait and host read. No CPU-device fallback.
"""

import ctypes as ct
import ctypes.util
import hashlib
import threading
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg


SHADER = """#version 450
#extension GL_ARB_gpu_shader_fp64 : require
layout(local_size_x = 128) in;
layout(std430, binding=0) readonly buffer Rows { int rows[]; };
layout(std430, binding=1) readonly buffer Cols { int cols[]; };
layout(std430, binding=2) readonly buffer Vals { double vals[]; };
layout(std430, binding=3) readonly buffer Input { double x[]; };
layout(std430, binding=4) writeonly buffer Output { double y[]; };
layout(push_constant) uniform Size { uint n; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    precise double total = 0.0;
    for (int j = rows[i]; j < rows[i+1]; ++j)
        total += vals[j] * x[cols[j]];
    y[i] = total;
}
"""


def compile_shader():
    """Compile our shader with the system shaderc C API (compute enum = 2)."""
    name = ctypes.util.find_library("shaderc_shared")
    if not name:
        raise RuntimeError("Vulkan experiment requires the shaderc shared library")
    lib = ct.CDLL(name)
    signatures = {
        "shaderc_compiler_initialize": (ct.c_void_p, []),
        "shaderc_compiler_release": (None, [ct.c_void_p]),
        "shaderc_compile_into_spv": (ct.c_void_p, [ct.c_void_p, ct.c_char_p,
            ct.c_size_t, ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_void_p]),
        "shaderc_result_get_compilation_status": (ct.c_int, [ct.c_void_p]),
        "shaderc_result_get_error_message": (ct.c_char_p, [ct.c_void_p]),
        "shaderc_result_get_length": (ct.c_size_t, [ct.c_void_p]),
        "shaderc_result_get_bytes": (ct.c_void_p, [ct.c_void_p]),
        "shaderc_result_release": (None, [ct.c_void_p]),
    }
    for symbol, (restype, argtypes) in signatures.items():
        function = getattr(lib, symbol)
        function.restype, function.argtypes = restype, argtypes
    compiler = lib.shaderc_compiler_initialize()
    if not compiler:
        raise RuntimeError("shaderc initialization failed")
    result = None
    try:
        source = SHADER.encode()
        result = lib.shaderc_compile_into_spv(
            compiler, source, len(source), 2, b"crystalab_spmv.comp", b"main", None
        )
        if not result or lib.shaderc_result_get_compilation_status(result) != 0:
            message = lib.shaderc_result_get_error_message(result) if result else b"no result"
            raise RuntimeError(f"shaderc compilation failed: {message!r}")
        return ct.string_at(lib.shaderc_result_get_bytes(result), lib.shaderc_result_get_length(result))
    finally:
        if result:
            lib.shaderc_result_release(result)
        lib.shaderc_compiler_release(compiler)


def canonical_csr(system):
    """Validate before narrowing indices or sending bytes to a shader."""
    if np.iscomplexobj(system):
        raise ValueError("Vulkan sparse products require real matrices")
    matrix = csr_matrix(system, dtype=np.float64, copy=True)
    matrix.check_format(full_check=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    n, m = matrix.shape
    if n == 0 or n != m or max(n, matrix.nnz) > np.iinfo(np.int32).max:
        raise ValueError("Vulkan sparse product requires a nonempty square int32 CSR")
    if not np.isfinite(matrix.data).all():
        raise ValueError("Vulkan sparse products require finite coefficients")
    arrays = (matrix.indptr.astype(np.int32), matrix.indices.astype(np.int32), matrix.data)
    digest = hashlib.sha256(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    for array in arrays:
        digest.update(array.tobytes())
    return matrix, arrays, digest.hexdigest()


class VulkanSpMV:
    """Synchronous, bounded matrix cache on a selected actual Vulkan GPU.

    A context owns its queue and all resources. It must not be shared across
    threads; the lock serializes solves and protects command-buffer reuse.
    """

    def __init__(self, *, expected_device_name=None, max_buffer_bytes=256 * 1024**2):
        import vulkan as vk

        if type(max_buffer_bytes) is not int or max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be a positive integer")
        self.vk = vk
        self._cleanup = []
        self._lock = threading.Lock()
        self._cache = {}
        self._closed = False
        self.max_buffer_bytes = max_buffer_bytes
        self.stats = dict(matrices=0, solves=0, dispatches=0, uploaded_bytes=0,
                          downloaded_bytes=0, allocated_buffer_bytes=0,
                          product_wall_s=0.0, solve_wall_s=0.0)
        started = time.perf_counter()
        try:
            self._initialize(expected_device_name)
        except BaseException:
            self.close()
            raise
        self.setup_wall_s = time.perf_counter() - started

    def _own(self, resource, destroy):
        self._cleanup.append(lambda: destroy(resource))
        return resource

    def _initialize(self, expected):
        v = self.vk
        self.instance = self._own(v.vkCreateInstance(v.VkInstanceCreateInfo(
            pApplicationInfo=v.VkApplicationInfo(pApplicationName="CrystaLab hybrid SpMV",
                                                 apiVersion=v.VK_API_VERSION_1_0)), None),
            lambda x: v.vkDestroyInstance(x, None))
        candidates = []
        for physical in v.vkEnumeratePhysicalDevices(self.instance):
            props = v.vkGetPhysicalDeviceProperties(physical)
            if props.deviceType not in (v.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                                        v.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU):
                continue
            if expected is not None and props.deviceName != expected:
                continue
            features = v.vkGetPhysicalDeviceFeatures(physical)
            if not features.shaderFloat64:
                continue
            for index, family in enumerate(v.vkGetPhysicalDeviceQueueFamilyProperties(physical)):
                if family.queueCount and family.queueFlags & v.VK_QUEUE_COMPUTE_BIT:
                    candidates.append((physical, props, features, index))
                    break
        if len(candidates) != 1:
            raise RuntimeError("Select exactly one actual fp64 GPU; CPU/llvmpipe fallback is forbidden")
        self.physical, props, features, self.family = candidates[0]
        if props.limits.maxComputeWorkGroupInvocations < 128 or props.limits.maxComputeWorkGroupSize[0] < 128:
            raise RuntimeError("GPU cannot run the declared 128-thread workgroup")
        self._properties = props  # Keep the owner of the nested limits struct alive.
        self.limits = props.limits
        self.info = dict(name=props.deviceName, vendor_id=int(props.vendorID),
                         device_id=int(props.deviceID), device_type=int(props.deviceType),
                         driver_version=int(props.driverVersion), api_version=int(props.apiVersion),
                         queue_family=self.family, shader_float64=True,
                         binding_version="1.3.275.1", workgroup_size=128,
                         memory_model="host-visible coherent; shared UMA memory may also be CPU RAM")
        self.device = self._own(v.vkCreateDevice(self.physical, v.VkDeviceCreateInfo(
            pQueueCreateInfos=[v.VkDeviceQueueCreateInfo(queueFamilyIndex=self.family,
                                                       pQueuePriorities=[1.0])],
            pEnabledFeatures=v.VkPhysicalDeviceFeatures(shaderFloat64=True,
                                                       robustBufferAccess=features.robustBufferAccess)), None),
            lambda x: v.vkDestroyDevice(x, None))
        self.queue = v.vkGetDeviceQueue(self.device, self.family, 0)
        bindings = [v.VkDescriptorSetLayoutBinding(binding=i, descriptorType=v.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1, stageFlags=v.VK_SHADER_STAGE_COMPUTE_BIT) for i in range(5)]
        self.descriptor_layout = self._own(v.vkCreateDescriptorSetLayout(self.device,
            v.VkDescriptorSetLayoutCreateInfo(pBindings=bindings), None),
            lambda x: v.vkDestroyDescriptorSetLayout(self.device, x, None))
        self.pipeline_layout = self._own(v.vkCreatePipelineLayout(self.device,
            v.VkPipelineLayoutCreateInfo(pSetLayouts=[self.descriptor_layout],
                pPushConstantRanges=[v.VkPushConstantRange(stageFlags=v.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=4)]), None),
            lambda x: v.vkDestroyPipelineLayout(self.device, x, None))
        spirv = compile_shader()
        self.info.update(shader_sha256=hashlib.sha256(SHADER.encode()).hexdigest(),
                         spirv_sha256=hashlib.sha256(spirv).hexdigest())
        shader = v.vkCreateShaderModule(self.device, v.VkShaderModuleCreateInfo(codeSize=len(spirv), pCode=spirv), None)
        try:
            # The binding copies an embedded stage by value. Retain its owner so
            # the allocated pName string stays alive through pipeline creation.
            stage = v.VkPipelineShaderStageCreateInfo(stage=v.VK_SHADER_STAGE_COMPUTE_BIT,
                                                     module=shader, pName="main")
            self.pipeline = self._own(v.vkCreateComputePipelines(self.device, v.VK_NULL_HANDLE, 1,
                [v.VkComputePipelineCreateInfo(layout=self.pipeline_layout, stage=stage)], None)[0],
                lambda x: v.vkDestroyPipeline(self.device, x, None))
        finally:
            v.vkDestroyShaderModule(self.device, shader, None)
        self.command_pool = self._own(v.vkCreateCommandPool(self.device,
            v.VkCommandPoolCreateInfo(queueFamilyIndex=self.family), None),
            lambda x: v.vkDestroyCommandPool(self.device, x, None))

    def _buffer(self, data):
        v = self.vk
        size = len(data)
        if not size or size > self.limits.maxStorageBufferRange:
            raise ValueError("CSR buffer exceeds the supported nonzero storage range")
        buffer = self._own(v.vkCreateBuffer(self.device, v.VkBufferCreateInfo(
            size=size, usage=v.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sharingMode=v.VK_SHARING_MODE_EXCLUSIVE), None),
            lambda x: v.vkDestroyBuffer(self.device, x, None))
        requirements = v.vkGetBufferMemoryRequirements(self.device, buffer)
        if self.stats["allocated_buffer_bytes"] + requirements.size > self.max_buffer_bytes:
            raise MemoryError("Vulkan matrix cache exceeds the declared buffer budget")
        memory_properties = v.vkGetPhysicalDeviceMemoryProperties(self.physical)
        required = v.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | v.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        types = [i for i in range(memory_properties.memoryTypeCount)
                 if requirements.memoryTypeBits & (1 << i)
                 and memory_properties.memoryTypes[i].propertyFlags & required == required]
        if not types:
            raise RuntimeError("Experiment requires host-visible coherent buffer memory")
        memory = self._own(v.vkAllocateMemory(self.device, v.VkMemoryAllocateInfo(
            allocationSize=requirements.size, memoryTypeIndex=types[0]), None),
            lambda x: v.vkFreeMemory(self.device, x, None))
        # Free bound memory only after destroying its buffer (LIFO cleanup).
        self._cleanup[-2], self._cleanup[-1] = self._cleanup[-1], self._cleanup[-2]
        v.vkBindBufferMemory(self.device, buffer, memory, 0)
        mapped = v.vkMapMemory(self.device, memory, 0, size, 0)
        self._cleanup.append(lambda: v.vkUnmapMemory(self.device, memory))
        mapped[:] = data
        self.stats["allocated_buffer_bytes"] += int(requirements.size)
        self.stats["uploaded_bytes"] += size
        return buffer, mapped, size

    def _prepare(self, matrix, arrays, key):
        if key in self._cache:
            return self._cache[key]
        v, n = self.vk, matrix.shape[0]
        if (n + 127) // 128 > self.limits.maxComputeWorkGroupCount[0]:
            raise ValueError("Matrix exceeds the declared dispatch limit")
        # A zero-nnz matrix still needs a valid nonzero descriptor allocation.
        payloads = [a.tobytes() or bytes(a.dtype.itemsize) for a in arrays] + [bytes(8*n), bytes(8*n)]
        buffers = [self._buffer(p) for p in payloads]
        pool = self._own(v.vkCreateDescriptorPool(self.device, v.VkDescriptorPoolCreateInfo(
            maxSets=1, pPoolSizes=[v.VkDescriptorPoolSize(type=v.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=5)]), None),
            lambda x: v.vkDestroyDescriptorPool(self.device, x, None))
        descriptor = v.vkAllocateDescriptorSets(self.device, v.VkDescriptorSetAllocateInfo(
            descriptorPool=pool, pSetLayouts=[self.descriptor_layout]))[0]
        writes = [v.VkWriteDescriptorSet(dstSet=descriptor, dstBinding=i,
            descriptorType=v.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[v.VkDescriptorBufferInfo(buffer=b, offset=0, range=size)])
            for i, (b, _, size) in enumerate(buffers)]
        v.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)
        command = v.vkAllocateCommandBuffers(self.device, v.VkCommandBufferAllocateInfo(
            commandPool=self.command_pool, level=v.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
        v.vkBeginCommandBuffer(command, v.VkCommandBufferBeginInfo())
        v.vkCmdPipelineBarrier(command, v.VK_PIPELINE_STAGE_HOST_BIT, v.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, [v.VkMemoryBarrier(srcAccessMask=v.VK_ACCESS_HOST_WRITE_BIT, dstAccessMask=v.VK_ACCESS_SHADER_READ_BIT)], 0, None, 0, None)
        v.vkCmdBindPipeline(command, v.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        v.vkCmdBindDescriptorSets(command, v.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout, 0, 1, [descriptor], 0, None)
        size_ptr = v.ffi.new("uint32_t *", n)
        v.vkCmdPushConstants(command, self.pipeline_layout, v.VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, size_ptr)
        v.vkCmdDispatch(command, (n + 127)//128, 1, 1)
        v.vkCmdPipelineBarrier(command, v.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, v.VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, [v.VkMemoryBarrier(srcAccessMask=v.VK_ACCESS_SHADER_WRITE_BIT, dstAccessMask=v.VK_ACCESS_HOST_READ_BIT)], 0, None, 0, None)
        v.vkEndCommandBuffer(command)
        fence = self._own(v.vkCreateFence(self.device, v.VkFenceCreateInfo(), None),
                          lambda x: v.vkDestroyFence(self.device, x, None))
        self._cache[key] = (buffers, command, fence, n)
        self.stats["matrices"] += 1
        return self._cache[key]

    def _matvec(self, prepared, vector):
        v = self.vk
        buffers, command, fence, n = prepared
        if np.iscomplexobj(vector):
            raise ValueError("Vulkan input vector must be real")
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (n,) or not np.isfinite(vector).all():
            raise ValueError("Vulkan input vector must be finite and match the matrix")
        started = time.perf_counter()
        buffers[3][1][:] = vector.tobytes()
        v.vkResetFences(self.device, 1, [fence])
        v.vkQueueSubmit(self.queue, 1, [v.VkSubmitInfo(pCommandBuffers=[command])], fence)
        v.vkWaitForFences(self.device, 1, [fence], True, 30_000_000_000)
        answer = np.frombuffer(buffers[4][1], dtype=np.float64, count=n).copy()
        self.stats["dispatches"] += 1
        self.stats["uploaded_bytes"] += vector.nbytes
        self.stats["downloaded_bytes"] += answer.nbytes
        self.stats["product_wall_s"] += time.perf_counter() - started
        if not np.isfinite(answer).all():
            raise ArithmeticError("GPU sparse product returned nonfinite values")
        return answer

    def matvec(self, system, vector):
        with self._lock:
            self._check_open()
            matrix, arrays, key = canonical_csr(system)
            return self._matvec(self._prepare(matrix, arrays, key), vector)

    def solve(self, system, rhs, guess):
        with self._lock:
            self._check_open()
            started = time.perf_counter()
            matrix, arrays, key = canonical_csr(system)
            prepared = self._prepare(matrix, arrays, key)
            operator = LinearOperator(matrix.shape, matvec=lambda x: self._matvec(prepared, x), dtype=np.float64)
            diagonal = matrix.diagonal()
            if np.any(diagonal <= 0):
                raise ValueError("SPD experiment requires a positive diagonal")
            preconditioner = LinearOperator(matrix.shape, matvec=lambda x: x / diagonal, dtype=np.float64)
            answer, status = cg(operator, rhs, x0=guess, M=preconditioner,
                                rtol=2e-13, atol=0.0, maxiter=4000)
            self.stats["solves"] += 1
            self.stats["solve_wall_s"] += time.perf_counter() - started
            return answer, status

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Vulkan context is closed")

    def close(self):
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "device"):
            self.vk.vkDeviceWaitIdle(self.device)
        for cleanup in reversed(self._cleanup):
            cleanup()
        self._cleanup.clear()
        self._cache.clear()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
