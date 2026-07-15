import asyncio
import json
import logging
import os
from typing import Any

from src.config import settings
from src.dcr_backends.factory import get_dcr_backend


def get_last_data_prov_event(log_json: list[dict[str, Any]]) -> Any | None:
    for event in reversed(log_json):
        if "dataset has been provisioned" in event["desc"]:
            return event["timestamp"]
    return None


async def run_periodic_monitoring() -> None:
    backend = get_dcr_backend()
    if not (
        backend.capabilities.supports_audit_log
        and backend.capabilities.supports_computation_output
    ):
        logging.info(
            "Periodic monitoring disabled for DCR provider %s: required capabilities are unavailable.",
            backend.provider_name,
        )
        return

    service_user = {"email": settings.decentriq_email or settings.local_auth_email}
    monitoring_config_path = os.path.join(settings.data_folder, "monitoring_config.json")

    print("run_periodic_monitoring called. Will wait 30 minutes before first check!")
    await asyncio.sleep(1800)
    while True:
        try:
            with open(monitoring_config_path) as config_file:
                monitoring = json.load(config_file)
                file_changed = False
                recent_provisions = monitoring["DCRs_most_recent_data_provisions"]
                for dcr_id in monitoring["DCRs_to_monitor"]:
                    print(f"\nchecking the log of DCR {dcr_id}...")
                    dcr_log = await backend.audit_log(dcr_id, service_user)
                    provisioned_at = get_last_data_prov_event(dcr_log)
                    if provisioned_at is None:
                        print("no data provision event found")
                        continue

                    if dcr_id not in recent_provisions or provisioned_at > recent_provisions[dcr_id]:
                        print("recent data provision found. Will run computations")
                        await backend.computation_output(dcr_id, service_user)
                        recent_provisions[dcr_id] = provisioned_at
                        print("Computations for ", dcr_id, " ran and saved.")
                        file_changed = True
                    else:
                        print(
                            "last provision_ts: ",
                            provisioned_at,
                            " last observed data provision_ts: ",
                            recent_provisions[dcr_id],
                        )
                time_between_checks = monitoring["time_between_checks"]

            if file_changed:
                with open(monitoring_config_path, "w") as config_file:
                    json.dump(monitoring, config_file, indent=4)
            print(f"Next check will be in {time_between_checks} seconds")
            await asyncio.sleep(time_between_checks)
        except FileNotFoundError:
            print("Error: The monitoring config file does not exist.")
            break
