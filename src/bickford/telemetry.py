# telemetry.py
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

def setup_tracing() -> None:
    # 1. Register Phoenix as OTEL backend & set global tracer provider
    tracer_provider = register(
        project_name="bickford-dev",
        set_global_tracer_provider=True,
        auto_instrument=False,  # we’ll be explicit
        batch=True,             # nice for production-ish use
    )

    # 2. Instrument OpenAI SDK
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    # 3. Instrument smolagents
    SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)