# FILE PURPOSE: Verify causal temperature feedback, bounded commands and analytic plant behavior.
# Independent algebraic cases cover filtered derivatives, anti-windup, recovery,
# delayed application and rejected command identities without using real hardware.

from dataclasses import replace
import math
import unittest

from controllers.contracts import ActuatorLimits, CommandProposal, ScalarObservation
from controllers.pid import PIDConfig, TemperaturePID
from controllers.thermal import SimulatedActuator, ThermalPlant


def observation(time=0, value=300, **changes):
    fields = dict(
        run_id="run",
        sample_id=f"sensor:{time}",
        channel="temperature",
        value=value,
        unit="K",
        measured_at_s=time,
        received_at_s=time,
        clock_id="clock",
        source="test_fixture",
    )
    fields.update(changes)
    return ScalarObservation(**fields)


def limits(**changes):
    fields = dict(
        channel="bath_temperature",
        unit="K",
        minimum=270.0,
        maximum=330.0,
        maximum_rate_per_s=100.0,
        provenance="TEST FIXTURE ONLY",
    )
    fields.update(changes)
    return ActuatorLimits(**fields)


def proposal(index, time=0, value=300, **changes):
    fields = dict(
        command_id=f"command:{index}",
        run_id="run",
        channel="bath_temperature",
        value=value,
        unit="K",
        proposed_at_s=time,
        expires_at_s=time + 10,
        clock_id="clock",
        evidence_ids=("test_evidence",),
        source="test_fixture",
    )
    fields.update(changes)
    return CommandProposal(**fields)


class PIDTests(unittest.TestCase):
    def controller(self, bound=None, **changes):
        config = replace(
            PIDConfig(initial_output_K=300, fallback_output_K=299), **changes
        )
        return TemperaturePID(config, bound or limits(), run_id="run", clock_id="clock")

    def test_proportional_only_and_initial_hold_have_independent_expected_values(self):
        pid = self.controller(kp=1.5, ki_per_s=0, kd_s=0)
        self.assertEqual(pid.update(0, 300, observation(value=298))["output_K"], 300)
        result = pid.update(1, 300, observation(1, 298))
        self.assertEqual(result["output_K"], 303)
        self.assertEqual(result["proportional_K"], 3)
        self.assertEqual(result["integral_after_K"], 0)

    def test_integral_accumulates_actual_time_without_future_measurements(self):
        pid = self.controller(kp=0, ki_per_s=1, kd_s=0)
        pid.update(0, 300, observation(value=299))
        self.assertEqual(pid.update(1, 300, observation(1, 299))["output_K"], 301)
        self.assertEqual(pid.update(3, 300, observation(3, 299))["output_K"], 303)

    def test_derivative_uses_measurement_not_setpoint_and_filter_decays(self):
        pid = self.controller(kp=0, ki_per_s=0, kd_s=2, derivative_filter_s=1)
        pid.update(0, 300, observation())
        result = pid.update(1, 300, observation(1, 301))
        self.assertAlmostEqual(result["derivative_K"], -2 * (1 - math.exp(-1)))
        result = pid.update(2, 310, observation(2, 301))
        self.assertAlmostEqual(
            result["derivative_K"], -2 * (1 - math.exp(-1)) * math.exp(-1)
        )
        self.assertLess(result["derivative_K"], 0)

    def test_saturation_rate_and_back_calculation_remain_bounded_and_recover(self):
        pid = self.controller(
            limits(minimum=295, maximum=305, maximum_rate_per_s=2),
            kp=2,
            ki_per_s=0.1,
            kd_s=0,
        )
        previous = 300
        for time in range(100):
            result = pid.update(time, 300, observation(time, 280))
            self.assertLessEqual(abs(result["output_K"] - previous), 2)
            self.assertTrue(295 <= result["output_K"] <= 305)
            self.assertTrue(-20 <= result["integral_after_K"] <= 20)
            previous = result["output_K"]
        self.assertLess(result["integral_after_K"], 0)
        reversed_error = pid.update(100, 300, observation(100, 310))
        self.assertLess(reversed_error["output_K"], 305)

    def test_held_sensor_packet_does_not_change_the_derivative_sampling_clock(self):
        pid = self.controller(kp=0, ki_per_s=0, kd_s=2, derivative_filter_s=1)
        pid.update(0, 300, observation())
        first = pid.update(1, 300, observation(1, 301))
        held = pid.update(2, 300, observation(1, 301, received_at_s=2))
        self.assertEqual(held["derivative_K"], first["derivative_K"])
        fresh = pid.update(3, 300, observation(3, 303))
        expected = math.exp(-2) * first["derivative_K"] + (1 - math.exp(-2)) * (-2)
        self.assertAlmostEqual(fresh["derivative_K"], expected)

    def test_missing_invalid_and_stale_feedback_freeze_integral_and_recover_without_jump(
        self,
    ):
        pid = self.controller(limits(maximum_rate_per_s=0.5))
        pid.update(0, 301, observation())
        first = pid.update(1, 301, None)
        self.assertEqual(first["mode"], "fallback")
        self.assertEqual(first["output_K"], 299.5)
        self.assertEqual(first["integral_before_K"], first["integral_after_K"])
        bad = observation(2, value=None, valid=False, reason="sensor disconnected")
        self.assertEqual(pid.update(2, 301, bad)["reason"], "invalid_feedback")
        recovered = pid.update(3, 301, observation(3, 300))
        self.assertEqual(recovered["mode"], "recovering")
        self.assertEqual(recovered["output_K"], 299)
        self.assertEqual(recovered["derivative_K"], 0)
        stale = pid.update(8, 301, observation(3, 300))
        self.assertEqual(stale["reason"], "stale_feedback")

    def test_duplicate_feedback_is_held_but_conflicting_or_out_of_order_data_is_rejected(
        self,
    ):
        pid = self.controller()
        pid.update(0, 300, observation())
        self.assertEqual(pid.update(1, 300, observation())["mode"], "tracking")
        self.assertEqual(
            pid.update(2, 300, observation(value=301))["reason"],
            "conflicting_feedback_timestamp",
        )
        pid.update(3, 300, observation(3))
        self.assertEqual(
            pid.update(4, 300, observation(2))["reason"], "out_of_order_feedback"
        )

    def test_run_clock_channel_and_delivery_mismatches_do_not_drive_feedback(self):
        cases = [
            observation(run_id="different"),
            observation(clock_id="different"),
            observation(channel="rotation", unit="rad/s"),
            observation(1),
        ]
        reasons = [
            "feedback_identity_mismatch",
            "feedback_identity_mismatch",
            "feedback_channel_mismatch",
            "feedback_not_yet_received",
        ]
        for sample, reason in zip(cases, reasons):
            with self.subTest(reason=reason):
                self.assertEqual(
                    self.controller().update(0, 300, sample)["reason"], reason
                )

    def test_manual_mode_tracks_bounded_output_and_release_is_bumpless(self):
        pid = self.controller()
        pid.update(0, 300, observation())
        manual = pid.update(1, 300, observation(1), manual_output_K=305)
        self.assertEqual(manual["mode"], "manual")
        self.assertEqual(manual["output_K"], 305)
        released = pid.update(2, 300, observation(2))
        self.assertEqual(released["mode"], "recovering")
        self.assertEqual(released["output_K"], 305)

    def test_update_gap_and_bad_arithmetic_leave_explicit_behavior(self):
        pid = self.controller()
        pid.update(0, 300, observation())
        self.assertEqual(
            pid.update(6, 300, observation(6))["reason"], "controller_update_gap"
        )
        self.assertEqual(pid.update(7, 300, observation(7))["mode"], "recovering")
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            pid.update(7, 300, observation(7))
        extreme = self.controller(kp=1e308)
        with self.assertRaisesRegex(ValueError, "finite"):
            extreme.update(0, 400, observation())
        self.assertIsNone(extreme.previous_time)


class ProcessContractTests(unittest.TestCase):
    def test_record_validation_preserves_units_times_and_missingness(self):
        self.assertEqual(
            observation(channel="rotation", unit="rad/s", value=-1).value, -1
        )
        for kwargs in [
            dict(unit="C"),
            dict(value=-1),
            dict(value=float("nan")),
            dict(received_at_s=-1),
            dict(received_at_s=0, measured_at_s=1),
            dict(value=None),
            dict(valid=False),
            dict(source=""),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                observation(**kwargs)
        for change in [
            dict(maximum=270),
            dict(maximum_rate_per_s=0),
            dict(provenance=""),
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                limits(**change)
        for change in [dict(ki_per_s=-1), dict(anti_windup_time_s=0), dict(kp=True)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(PIDConfig(), **change)

    def test_actuator_acceptance_and_fractional_application_are_separate(self):
        adapter = SimulatedActuator(
            limits(), run_id="run", clock_id="clock", delay_s=0.5, initial_value=300
        )
        receipt = adapter.submit(proposal(0), 0, authorization="simulation_policy")
        self.assertEqual(receipt["event"], "accepted")
        adapter.advance(0.5)
        adapter.submit(proposal(1, 1, 310), 1, authorization="simulation_policy")
        self.assertEqual(adapter.advance(1.49), [])
        self.assertEqual(adapter.applied_value, 300)
        applied = adapter.advance(2)
        self.assertEqual(applied[0]["time_s"], 1.5)
        self.assertEqual(adapter.applied_value, 310)
        with self.assertRaises(ValueError):
            adapter.advance(1)

    def test_rejected_commands_do_not_change_applied_or_accepted_values(self):
        adapter = SimulatedActuator(
            limits(maximum_rate_per_s=1),
            run_id="run",
            clock_id="clock",
            delay_s=2,
            initial_value=300,
        )
        cases = [
            (dict(run_id="other"), "simulation_policy", "command_identity_mismatch"),
            (
                dict(channel="rotation", unit="rad/s"),
                "simulation_policy",
                "command_channel_mismatch",
            ),
            ({}, "operator_approved", "simulation_authorization_required"),
            (dict(expires_at_s=1), "simulation_policy", "expires_before_application"),
            (dict(value=340), "simulation_policy", "command_outside_range"),
            (dict(value=305), "simulation_policy", "command_exceeds_rate"),
        ]
        for index, (changes, authorization, reason) in enumerate(cases):
            with self.subTest(reason=reason):
                response = adapter.submit(
                    proposal(index, **changes), 0, authorization=authorization
                )
                self.assertEqual(response["reason"], reason)
                self.assertEqual(adapter.applied_value, 300)
                self.assertEqual(adapter.last_accepted_value, 300)
        self.assertEqual(
            adapter.submit(proposal(0), 0, authorization="simulation_policy")["reason"],
            "duplicate_command_id",
        )
        self.assertEqual(adapter.pending, [])

    def test_thermal_solution_and_fractional_boundaries_match_independent_algebra(self):
        plant = ThermalPlant(
            initial_K=300,
            initial_input_K=300,
            time_constant_s=10,
            disturbances=[dict(time_s=0, offset_K=0), dict(time_s=0.5, offset_K=2)],
        )
        application = dict(
            event="applied", channel="bath_temperature", unit="K", time_s=1.5, value=310
        )
        # 0..0.5: 300; 0.5..1.5 equilibrium 302; 1.5..2 equilibrium 312.
        at_15 = 302 + (300 - 302) * math.exp(-1 / 10)
        expected = 312 + (at_15 - 312) * math.exp(-0.5 / 10)
        self.assertAlmostEqual(plant.advance(2, [application]), expected, places=12)
        self.assertEqual(plant.input_K, 310)
        with self.assertRaises(ValueError):
            plant.advance(3, [dict(application, time_s=1)])
