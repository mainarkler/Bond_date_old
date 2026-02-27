from __future__ import annotations

from dataclasses import dataclass

from analytics_app.application.services.sell_stress_service import SellStressService
from analytics_app.application.services.vm_calculation_service import VMCalculationService
from analytics_app.infrastructure.clients.cbr_client import CBRClient
from analytics_app.infrastructure.clients.moex_client import MOEXClient
from analytics_app.infrastructure.config import Settings, get_settings
from analytics_app.infrastructure.logging_config import setup_logging


@dataclass(frozen=True)
class Container:
    settings: Settings
    moex_client: MOEXClient
    cbr_client: CBRClient
    sell_stress_service: SellStressService
    vm_calculation_service: VMCalculationService


def build_container() -> Container:
    settings = get_settings()
    setup_logging(settings.log_level)

    moex_client = MOEXClient(settings)
    cbr_client = CBRClient(settings)

    return Container(
        settings=settings,
        moex_client=moex_client,
        cbr_client=cbr_client,
        sell_stress_service=SellStressService(market_data_client=moex_client),
        vm_calculation_service=VMCalculationService(),
    )
