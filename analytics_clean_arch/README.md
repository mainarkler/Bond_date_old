# analytics-clean-arch

Scaffold of a new Python 3.11+ analytics project using Clean/Hexagonal architecture principles.

## Goals

- Separate **presentation**, **application**, **domain**, and **infrastructure** concerns.
- Keep the core business logic framework-agnostic.
- Make later attachment to Streamlit/FastAPI/Django straightforward.

## Architecture

```text
presentation/   -> entrypoints (CLI now; Streamlit/FastAPI/Django adapters later)
application/    -> orchestration services + use-case level error handling
domain/         -> pure financial logic and domain exceptions
infrastructure/ -> external APIs, logging, configuration, DI wiring
schemas/        -> Pydantic DTOs for validation and typed boundaries
tests/          -> unit tests for domain formulas and services
```

### Layer rules

- `domain` imports nothing from web/UI/frameworks.
- `application` depends on `domain` and contracts, not concrete infrastructure.
- `infrastructure` implements contracts (clients, adapters).
- `presentation` calls application services only.

## Included examples

- `SellStressService`
- `VMCalculationService`
- `MOEXClient` with retry logic
- Pydantic request/response DTOs
- Unit test for sell stress formula
- Manual dependency injection container

## Run

```bash
make install
make run
```

## Test

```bash
make test
```

## Error handling strategy

- Domain raises deterministic `DomainError` subclasses.
- Infrastructure wraps transport failures into `InfrastructureError`.
- Application maps lower-level errors to `ApplicationServiceError` for stable presentation behavior.

## Future adapters

You can add:

- `presentation/streamlit_app.py` for interactive dashboard UI.
- `presentation/api/` for FastAPI endpoints.
- `presentation/django_app/` for Django views + DRF.

Core services and domain formulas remain unchanged.
