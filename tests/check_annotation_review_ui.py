# FILE PURPOSE: Exercise the local editor in an optional offscreen Qt browser using fixture data.
# A real bundle, when supplied, is served read-only and must remain byte-identical.
"""Standalone browser QA; PySide6 is a development tool, not a UI runtime dependency."""

import argparse
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import sys
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6 import __version__ as qt_binding_version
from PySide6.QtCore import QEventLoop, QPoint, QTimer, QUrl, Qt
from PySide6.QtTest import QTest
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from annotation_review_fixture import prepare_review_fixture
from dashboard.review import ReviewServer
from vision.video import file_sha256


def files_snapshot(root):
    return {
        str(p.relative_to(root)): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class Page(QWebEnginePage):
    def __init__(self, profile, parent):
        super().__init__(profile, parent)
        self.errors = []

    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            self.errors.append(dict(message=message, line=line, source=source))


class BrowserQA:
    def __init__(self, output):
        self.output = output
        self.view = QWebEngineView()
        # An unnamed profile is off-the-record; no personal browser profile is touched.
        self.profile = QWebEngineProfile(self.view)
        self.profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
        self.page = Page(self.profile, self.view)
        self.view.setPage(self.page)
        self.view.resize(1440, 1180)
        self.view.show()
        self.checks = []

    def evaluate(self, expression):
        loop = QEventLoop()
        result = []

        def receive(value):
            result.append(value)
            loop.quit()

        self.page.runJavaScript(
            "(()=>{try{return JSON.stringify({value:("
            + expression
            + ")})}catch(error){return JSON.stringify({qa_error:String(error)})}})()",
            receive,
        )
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(5000)
        loop.exec()
        timer.stop()
        if not result:
            raise TimeoutError("Browser evaluation timed out")
        decoded = {} if result[0] in (None, "") else json.loads(result[0])
        if "qa_error" in decoded:
            raise RuntimeError(decoded["qa_error"])
        return decoded.get("value")

    def wait(self, condition):
        for _ in range(100):
            try:
                if self.evaluate(condition):
                    return
            except RuntimeError:
                # Polling may overlap navigation; do not emit injected console errors.
                pass
            QTest.qWait(100)
        raise AssertionError("UI condition not reached: " + condition)

    def check(self, name, condition):
        self.wait(condition)
        self.checks.append(name)

    def click(self, element):
        self.evaluate(f"document.getElementById({json.dumps(element)}).click()")
        QTest.qWait(80)

    def input(self, element, value, event="input"):
        prop = "checked" if isinstance(value, bool) else "value"
        self.evaluate(
            f"(()=>{{const e=document.getElementById({json.dumps(element)});e.{prop}={json.dumps(value)};e.dispatchEvent(new Event({json.dumps(event)},{{bubbles:true}}));}})()"
        )

    def point(self, native):
        point = self.evaluate(
            f"(()=>{{const r=canvas.getBoundingClientRect();return [r.left+state.view.x+{native[0]}*state.view.scale,r.top+state.view.y+{native[1]}*state.view.scale];}})()"
        )
        return QPoint(round(point[0]), round(point[1]))

    def canvas_click(self, native):
        QTest.mouseClick(
            self.view.focusProxy(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self.point(native),
        )
        QTest.qWait(80)

    def screenshot(self, name):
        QTest.qWait(250)
        if not self.view.grab().save(str(self.output / name)):
            raise OSError("Cannot save browser screenshot")

    def open(self, server):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            if response.status != 200 or b"Check the crystal" not in response.read():
                raise AssertionError("Local page did not return HTTP 200")
        finally:
            connection.close()
        self.view.load(QUrl(f"http://127.0.0.1:{server.server_port}/"))
        self.wait(
            f"window.location.port==='{server.server_port}' && "
            "typeof state!=='undefined' && state.session && state.loaded && !state.busy"
        )

    def fixture(self, server):
        self.open(server)
        self.check(
            "Initial proposals are unreviewed; review/export disabled",
            "state.session.reviewed===0 && $('review').disabled && $('export').disabled && state.doc.shapes.length===1",
        )
        self.evaluate("(()=>{Object.assign(state.current.source,{phase_context:'Fixture phase <b>literal text</b>',caption_id:'S1',display_elapsed_s:1.25,published_speedup_relative_elapsed_s:17.5,license:'FIXTURE ONLY',source_url:'javascript:void(0)'});renderSourceContext();return true;})()")
        self.check(
            "Phase context is literal text and unsafe article links remain hidden",
            "!$('source-context').hidden && $('source-phase').textContent==='Fixture phase <b>literal text</b>' && !$('source-phase').querySelector('b') && $('source-paper').hidden",
        )
        self.check(
            "Source display and inferred elapsed time stay distinct from acquisition/run clocks",
            "$('source-clock').textContent.includes('Display 1.250 s') && $('source-clock').textContent.includes('17.500 s since clip start') && $('source-clock').textContent.includes('growth-run start unknown')",
        )
        self.evaluate("(()=>{for(const key of ['phase_context','caption_id','display_elapsed_s','published_speedup_relative_elapsed_s','license','source_url'])delete state.current.source[key];renderSourceContext();return true;})()")
        self.check("Sources without phase metadata retain the existing workspace", "$('source-context').hidden")
        self.screenshot("fixture_desktop.png")
        initial = self.evaluate("state.doc.shapes[0].points[0]")
        destination = [initial[0] + 8, initial[1] + 4]
        QTest.mousePress(
            self.view.focusProxy(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self.point(initial),
        )
        QTest.mouseMove(self.view.focusProxy(), self.point(destination), 120)
        QTest.mouseRelease(
            self.view.focusProxy(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self.point(destination),
        )
        self.check(
            "Native canvas drag changes a polygon vertex",
            f"dirty() && Math.abs(state.doc.shapes[0].points[0][0]-{destination[0]})<2",
        )
        self.click("undo")
        self.check(
            "Undo restores the dragged outline",
            f"JSON.stringify(state.doc.shapes[0].points[0])==={json.dumps(json.dumps(initial, separators=(',', ':')))}",
        )
        # The starting fixture silhouette is a quadrilateral; select an edge midpoint.
        points = self.evaluate("state.doc.shapes[0].points")
        midpoint = [(points[0][i] + points[1][i]) / 2 for i in (0, 1)]
        self.canvas_click(midpoint)
        self.check(
            "Clicking an edge inserts a vertex",
            f"state.doc.shapes[0].points.length==={len(points) + 1}",
        )
        QTest.keyClick(self.view.focusProxy(), Qt.Key.Key_Delete)
        self.check(
            "Delete removes the selected vertex",
            f"state.doc.shapes[0].points.length==={len(points)}",
        )
        self.click("draw")
        self.click("fit-image")
        for point in ([480, 280], [535, 280], [515, 330]):
            self.canvas_click(point)
        self.click("finish")
        self.check(
            "New outline preserves an explicit object class",
            "state.doc.shapes.length===2 && state.doc.shapes[1].label==='secondary_crystal'",
        )
        self.input("shape-uncertain", True, "change")
        self.input("reviewer", "AUTOMATED UI FIXTURE ONLY")
        self.input("ack", True, "change")
        self.check("An uncertain object disables review", "$('review').disabled")
        self.click("save")
        self.check(
            "Uncertain draft saves without a receipt",
            "!state.busy && !dirty() && state.current.review===null && state.doc.shapes[1].flags.uncertain",
        )
        self.click("remove-shape")
        self.input("image-uncertain", True, "change")
        self.click("save")
        self.check(
            "Image-level uncertainty stays a draft",
            "!state.busy && state.doc.flags.uncertain && state.session.reviewed===0 && $('review').disabled",
        )
        self.input("image-uncertain", False, "change")
        self.input("ack", True, "change")
        self.input("notes", "Automated fixture review; no real human label")
        self.check(
            "Editing after acknowledgment clears it",
            "!$('ack').checked && $('review').disabled",
        )
        self.input("ack", True, "change")
        self.click("review")
        self.check(
            "Explicit fixture review saves and locks the exact labels",
            "!state.busy && state.session.reviewed===1 && state.current.status==='reviewed' && $('save').disabled && $('edit').disabled",
        )
        self.click("export")
        self.check(
            "Export includes only reviewed fixture image",
            "!state.busy && $('message').textContent.includes('Exported 1 reviewed') && $('message').textContent.includes('2 unreviewed')",
        )
        before_revision = self.evaluate("state.session.bundle_path")
        self.click("revise")
        self.check(
            "Revision unlocks selected labels and removes its prior acknowledgment",
            f"!state.busy && state.current.review===null && !$('ack').checked && state.session.bundle_path!=={json.dumps(before_revision)} && !$('edit').disabled",
        )
        self.click("next")
        self.check(
            "Navigation clears review acknowledgment",
            "!state.busy && state.index===1 && !$('ack').checked && state.loaded",
        )
        self.click("remove-shape")
        self.input("ack", True, "change")
        self.check(
            "Empty image needs separate explicit negative acknowledgment",
            "!$('empty-row').hidden && $('review').disabled",
        )
        self.input("empty-reviewed", True, "change")
        self.click("review")
        self.check(
            "Reviewed empty fixture has an explicit negative receipt",
            "!state.busy && state.current.review?.empty_reviewed===true",
        )
        self.view.resize(390, 844)
        QTest.qWait(300)
        self.check(
            "Mobile layout fits without horizontal page overflow",
            "document.documentElement.scrollWidth<=window.innerWidth+1",
        )
        self.screenshot("fixture_mobile.png")
        if self.page.errors:
            raise AssertionError(self.page.errors)

    def real_preview(self, server):
        self.view.resize(1440, 1180)
        self.open(server)
        self.check(
            "Real bundle opens read-only with zero review receipts",
            "state.session.read_only && state.session.samples.length===6 && state.session.reviewed===0 && $('review').disabled && $('export').disabled && $('save').disabled",
        )
        self.screenshot("real_read_only_desktop.png")
        self.click("fit-image")
        self.check(
            "Original real image dimensions remain native",
            "state.image.naturalWidth===state.current.source.image_width && state.image.naturalHeight===state.current.source.image_height",
        )
        if self.evaluate("Boolean(state.current.source.phase_context)"):
            self.check(
                "Real phase, license and unknown physical clock are displayed",
                "!$('source-context').hidden && $('source-phase').textContent===state.current.source.phase_context && $('source-provenance').textContent.includes(state.current.source.license) && $('source-clock').textContent.includes('growth-run start unknown')",
            )
        self.screenshot("real_read_only_full_image.png")
        self.click("next")
        self.check(
            "Read-only preview can navigate real proposals",
            "!state.busy && state.index===1 && state.loaded && state.current.review===null",
        )
        if self.evaluate("Boolean(state.current.source.phase_context)"):
            for index in range(2,self.evaluate("state.session.samples.length")):
                self.click("next")
                self.check(
                    f"Native source context follows real frame {index}",
                    f"!state.busy && state.index==={index} && state.loaded && $('source-phase').textContent===state.current.source.phase_context && state.image.naturalWidth===state.current.source.image_width && $('review').disabled",
                )
            self.view.resize(390,844)
            self.check("Real source context fits the mobile viewport", "document.documentElement.scrollWidth<=window.innerWidth+1")
            self.screenshot("real_read_only_mobile.png")
        if self.page.errors:
            raise AssertionError(self.page.errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-bundle", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    app = QApplication([])
    browser = BrowserQA(args.output)
    servers = []
    threads = []
    report = dict(
        schema="crystalab_annotation_ui_qa_v1",
        pyside6=qt_binding_version,
        status="running",
        actual_human_reviews_created=0,
        checks=browser.checks,
    )

    def start(bundle, read_only=False):
        server = ReviewServer(bundle, port=0, read_only=read_only)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        threads.append(thread)
        return server

    try:
        browser.fixture(start(prepare_review_fixture(args.output)))
        if args.real_bundle:
            before = files_snapshot(args.real_bundle)
            browser.real_preview(start(args.real_bundle, read_only=True))
            after = files_snapshot(args.real_bundle)
            report["real_bundle"] = dict(
                path=str(args.real_bundle.resolve()),
                unchanged=before == after,
                before=before,
                after=after,
            )
            if before != after:
                raise AssertionError("Read-only preview changed real bundle files")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        browser.screenshot("failure.png")
        raise
    finally:
        report["console_errors"] = browser.page.errors
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=3)
        report["servers_stopped"] = all(not t.is_alive() for t in threads)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        browser.view.close()
        app.processEvents()
    print(
        json.dumps(
            dict(
                status=report["status"],
                checks=len(browser.checks),
                report=str(args.output / "report.json"),
                servers_stopped=report["servers_stopped"],
            )
        )
    )


if __name__ == "__main__":
    main()
