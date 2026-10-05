from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import auth, ml, predictions, system
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.database.session import Database
from app.ml.factory import make_dataset
from app.middleware.request_logging import RequestLoggingMiddleware
from app.services.model_service import ModelService
from app.services.training_service import TrainingService

API_VERSION = "0.2.0"
DESCRIPTION = (
    "Research prototype for estimating self-reported affect categories (e.g. low vs high valence) "
    "from EEG recordings of the DEAP dataset. Not a medical or diagnostic device."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_dir)
    logger = get_logger("app")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.ensure_output_dirs()
        logger.info("Starting NeuroSense API", extra={"environment": settings.environment, "db": app.state.db.dialect})
        app.state.model_service.load_active()
        if app.state.model_service.available:
            try:
                with app.state.db.session() as session:
                    app.state.model_service.sync_to_db(session)
                    session.commit()
            except Exception as exc:
                logger.error("Could not sync active model to database", extra={"error": str(exc)})
        yield
        app.state.training_service.wait(timeout=1)
        app.state.db.dispose()
        logger.info("NeuroSense API stopped")

    app = FastAPI(title="NeuroSense AI API", version=API_VERSION, description=DESCRIPTION, lifespan=lifespan,
                  docs_url=f"{settings.api_prefix}/docs", redoc_url=f"{settings.api_prefix}/redoc",
                  openapi_url=f"{settings.api_prefix}/openapi.json",
                  swagger_ui_oauth2_redirect_url=f"{settings.api_prefix}/docs/oauth2-redirect")
    app.state.settings = settings
    app.state.db = Database(settings.database_url)
    app.state.dataset = make_dataset(settings)
    app.state.model_service = ModelService(settings)
    app.state.training_service = TrainingService(settings, app.state.db, app.state.model_service)

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                       allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type", "X-Request-ID"])
    register_exception_handlers(app)

    api = APIRouter(prefix=settings.api_prefix)
    for module in (system, auth, ml, predictions):
        api.include_router(module.router)
    app.include_router(api)

    settings.graphs_dir.mkdir(parents=True, exist_ok=True)
    app.mount(f"{settings.api_prefix}/graphs", StaticFiles(directory=settings.graphs_dir), name="graphs")
    return app
