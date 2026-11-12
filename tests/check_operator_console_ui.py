# FILE PURPOSE: Browser-check replay review, simulated pause/resume and actual read-only sources.
# Writable interactions use generated fixture pixels and automated-QA receipts only.
import argparse
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import sys
import threading
import subprocess
import selectors
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtCore import QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from check_annotation_review_ui import BrowserQA, files_snapshot
from test_operator_console import prepare_operator_fixture
from test_operator_forecast import prepare_forecast_fixture
from dashboard.operator import OperatorServer, OperatorStore
from controllers.audit import audit_control_benchmark


class OperatorBrowserQA(BrowserQA):
    def open(self, server):
        c = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            c.request("GET", "/")
            r = c.getresponse()
            assert r.status == 200 and b"Crystal growth, in view" in r.read()
        finally:
            c.close()
        self.view.load(QUrl(f"http://127.0.0.1:{server.server_port}/"))
        self.wait(
            f"window.location.port==='{server.server_port}' && typeof operatorState!=='undefined' && operatorState.session && operatorState.frame && !operatorState.busy && $('frame-image').naturalWidth>0"
        )

    def seek(self, index):
        self.input("timeline", str(index))
        self.wait(
            f"operatorState.index==={index} && operatorState.frame.frame.ordinal==={index} && !operatorState.loadingFrame && $('frame-image').dataset.ordinal==='{index}' && $('frame-image').complete"
        )

    def fixture(self, server):
        self.open(server)
        forecast = server.store.forecast is not None
        duration = (
            60
            if not forecast
            else server.store.frame(13)["forecast_context"]["recommendation"][
                "pause_model_s"
            ]
        )
        self.check(
            "Empty operator history and unavailable startup sensor",
            "operatorState.session.reviews.length===0 && $('sensor').textContent==='Unavailable' && $('approve').disabled",
        )
        self.click("play")
        self.check("Play advances recorded source frames", "operatorState.index>0")
        self.click("play")
        self.seek(13)
        self.evaluate(
            "Array.from(document.querySelectorAll('.proposal')).find(e=>e.textContent.includes('Review ramp pause')).click()"
        )
        self.check(
            "Review remains explicit at a fresh ramp proposal",
            "operatorState.frame.proposals.some(p=>p.can_approve) && $('approve').disabled",
        )
        if forecast:
            self.check(
                "Source-bound adaptive state and area forecast are displayed",
                "operatorState.frame.forecast_context.state.source_frame_index===13 && $('forecast-area').textContent.includes('px²') && $('forecast-recommendation').textContent.includes('Suggested pause') && $('forecast-reference').textContent.includes(operatorState.frame.forecast_context.context_sha256)",
            )
            self.input("pause-duration", "61")
        self.screenshot("fixture_before_review.png")
        self.input("reviewer", "Automated browser fixture check")
        self.input("acknowledge-check", True)
        if forecast:
            self.check(
                "Out-of-bounds requested duration keeps approval blocked",
                "$('approve').disabled",
            )
            self.click("use-suggestion")
            self.check(
                "Using a suggestion fills its exact duration and requires acknowledgement",
                f"Number($('pause-duration').value)==={duration} && operatorState.pauseBasis==='adaptive_suggestion' && !$('acknowledge-check').checked && $('approve').disabled",
            )
            self.input("acknowledge-check", True)
        self.check(
            "Named acknowledgement enables simulated pause", "!$('approve').disabled"
        )
        self.evaluate(
            "(()=>{window.__qaDecode=Image.prototype.decode;Image.prototype.decode=function(){return window.__qaDecode.call(this).then(()=>new Promise(resolve=>setTimeout(resolve,500)));};return true;})()"
        )
        self.input("timeline", "14")
        self.check(
            "Delayed frame pixels keep review blocked and the displayed clock unchanged",
            "operatorState.loadingFrame && $('approve').disabled && operatorState.index===13 && $('frame-image').dataset.ordinal==='13'",
        )
        self.wait(
            "operatorState.index===14 && !operatorState.loadingFrame && $('frame-image').dataset.ordinal==='14'"
        )
        self.evaluate("Image.prototype.decode=window.__qaDecode")
        self.seek(13)
        if forecast:
            self.click("use-suggestion")
        self.input("acknowledge-check", True)
        self.check(
            "Review re-enables only after selected frame and evidence decode",
            "!$('approve').disabled && operatorState.evidenceReady===operatorState.selected",
        )
        self.click("approve")
        self.check(
            "Saved automated receipt changes the bounded simulation",
            f"!operatorState.busy && operatorState.session.reviews.length===1 && operatorState.session.reviews[0].reviewer_kind==='automated_fixture_qa' && operatorState.session.runs[2].recipe_phase_lag_s==={duration}",
        )
        if forecast:
            self.check(
                "Receipt preserves the displayed forecast and selected suggestion",
                "operatorState.session.reviews[0].forecast_context.context_sha256===operatorState.frame.forecast_context.context_sha256 && operatorState.session.reviews[0].pause_basis==='adaptive_suggestion'",
            )
        self.check(
            "Accepted request, applied input and delayed response stay separate",
            "operatorState.frame.process.dispatch.effective_at_s===operatorState.frame.process.time_s+2 && operatorState.frame.process.supervision.pause_active && $('resume').disabled",
        )
        self.seek(15)
        self.check(
            "Moving to another frame clears review acknowledgement",
            "!$('acknowledge-check').checked && $('resume').disabled",
        )
        self.input("acknowledge-check", True)
        self.check("Active pause can be manually resumed", "!$('resume').disabled")
        self.click("resume")
        self.check(
            "Manual resume produces a new immutable result",
            "!operatorState.busy && operatorState.session.reviews.length===2 && operatorState.session.runs[2].recipe_phase_lag_s===20 && !operatorState.frame.process.supervision.pause_active",
        )
        self.screenshot("fixture_after_resume.png")
        self.seek(0)
        self.check(
            "Seeking before saved decisions cannot create retrospective approval",
            "operatorState.frame.proposals.every(p=>!p.can_review) && $('approve').disabled && $('resume').disabled",
        )
        proposal = self.evaluate(
            "operatorState.session.proposals.find(p=>p.proposal.kind==='inclusion_alert').proposal"
        )
        index = self.evaluate(
            f"operatorState.session.frames.findIndex(f=>f.time_s>={proposal['proposed_at_s']})"
        )
        self.seek(index)
        self.evaluate(
            "Array.from(document.querySelectorAll('.proposal')).find(e=>e.textContent.includes('Inspect possible inclusion')).click()"
        )
        self.check(
            "Localized inclusion evidence is displayed",
            "$('evidence-image').naturalWidth>0 && operatorState.evidenceReady===operatorState.selected && $('proposal-title').textContent.includes('inclusion')",
        )
        self.view.resize(390, 1000)
        QTest.qWait(300)
        self.check(
            "Mobile document has no horizontal page overflow",
            "document.documentElement.scrollWidth<=window.innerWidth+1",
        )
        self.screenshot("fixture_mobile.png")
        self.view.resize(1440, 1180)
        history, _ = server.store.history()
        assert (
            audit_control_benchmark(server.store.root / history["latest_run"])["status"]
            == "pass"
        )
        return dict(
            session=str(server.store.root),
            reviews=2,
            human_reviews=0,
            requested_pause_model_s=duration,
            final_recipe_lag_model_s=20,
        )

    def readonly(self, session, name):
        before = files_snapshot(session)
        server = OperatorServer(session, port=0, read_only=True)
        source_before = files_snapshot(server.store.source.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.open(server)
            self.check(
                f"{name}: source and controls remain read-only",
                "operatorState.session.read_only && operatorState.session.reviews.length===0 && $('approve').disabled && $('acknowledge').disabled && $('read-only').hidden===false",
            )
            self.seek(len(server.store.frames) // 2)
            self.check(
                f"{name}: selected frame and proposal pixels finish decoding",
                "!operatorState.loadingFrame && (!operatorState.selected || operatorState.evidenceReady===operatorState.selected)",
            )
            if server.store.forecast is not None:
                self.check(
                    f"{name}: forecast matches the selected source frame",
                    "operatorState.frame.forecast_context.source_frame_index===operatorState.frame.frame.frame_index && $('forecast-reference').textContent.includes(operatorState.frame.forecast_context.context_sha256) && !$('forecast-content').hidden",
                )
            if not server.store.manifest["simulation_available"]:
                self.check(
                    f"{name}: process readings remain absent",
                    "operatorState.session.chart===null && !$('process-empty').hidden && $('process-content').hidden",
                )
                if server.store.forecast is not None:
                    self.check(
                        f"{name}: real forecast stays uncalibrated shadow output",
                        "$('forecast-note').textContent.includes('uncalibrated') && $('pause-duration-controls').hidden && $('approve').hidden && $('resume').hidden",
                    )
            self.screenshot(name + "_readonly.png")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        assert before == files_snapshot(session)
        assert source_before == files_snapshot(server.store.source.root)
        server.store.source.verify()
        return dict(
            session=str(session),
            files_unchanged=len(before),
            source_files_unchanged=len(source_before),
            human_reviews_created=0,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--synthetic-session", type=Path)
    p.add_argument("--real-session", type=Path)
    p.add_argument("--forecast-fixture", action="store_true")
    a = p.parse_args()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = (
        prepare_forecast_fixture(output)
        if a.forecast_fixture
        else prepare_operator_fixture(output)
    )
    app = QApplication.instance() or QApplication([])
    qa = OperatorBrowserQA(output)
    log = (output / "server.log").open("w")
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/operator_console.py"),
        "--session",
        str(fixture),
        "--port",
        "0",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    if not selector.select(timeout=10):
        process.terminate()
        raise TimeoutError("Operator server did not print its local URL")
    line = process.stdout.readline()
    selector.close()
    if not line.startswith("CrystaLab operator: http://127.0.0.1:"):
        process.terminate()
        raise RuntimeError("Operator server failed to start: " + line)
    server = SimpleNamespace(
        server_port=int(line.strip().rsplit(":", 1)[1]), store=OperatorStore(fixture)
    )
    result = {}
    try:
        result["fixture"] = qa.fixture(server)
        for name, session in (
            ("synthetic", a.synthetic_session),
            ("real", a.real_session),
        ):
            if session is not None:
                result[name] = qa.readonly(session.resolve(), name)
        assert not qa.page.errors, qa.page.errors
        result.update(status="passed", checks=qa.checks, browser_errors=qa.page.errors)
        (output / "browser_report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    except Exception as error:
        qa.screenshot("failure.png")
        (output / "browser_failure.json").write_text(
            json.dumps(
                dict(
                    error=str(error),
                    checks=qa.checks,
                    browser_errors=qa.page.errors,
                    ui_message=qa.evaluate("$('message').textContent"),
                ),
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        process.terminate()
        process.wait(timeout=10)
        log.close()
        qa.view.close()
        app.processEvents()


if __name__ == "__main__":
    main()
