from src.config import settings
from src.decentriq import get_dcr_log, run_computation_get_output
import json
import asyncio


def get_last_data_prov_event(log_json):
    for e in reversed(log_json):
        if e['desc'].find("dataset has been provisioned")>-1:
            return e['timestamp']
    return None

          

async def run_periodic_monitoring():
    #waiting for the app to fully launch
    print("run_periodic_monitoring called. Will wait 30 minutes before first check!")
    await asyncio.sleep(1800)
    while True:
        try:
            with open("/data/monitoring_config.json", 'r') as f:
                ms = json.load(f)
                filechanged= False
                for dcr_id in ms['DCRs_to_monitor']:
                    print(f"\nchecking the log of DCR {dcr_id}...")
                    dcr_log = get_dcr_log(dcr_id)
                    prov_ts = get_last_data_prov_event(dcr_log)
                    if prov_ts == None:
                        print("no data provision event found")
                        continue
                    elif (prov_ts != None and 
                        (dcr_id not in ms['DCRs_most_recent_data_provisions'] or prov_ts>ms['DCRs_most_recent_data_provisions'][dcr_id])):
                        print("recent data provision found. Will run computations")
                        run_computation_get_output(dcr_id)
                        ms['DCRs_most_recent_data_provisions'][dcr_id] = prov_ts
                        print("Computations for ", dcr_id, " ran and saved.")
                        filechanged = True
                    else:  #daata provision event from decentriq log already seen
                        print("last provision_ts: ", prov_ts, " last observed data provision_ts: ", 
                            ms['DCRs_most_recent_data_provisions'][dcr_id])
                time_between_checks = ms['time_between_checks']
            if filechanged:
                with open(settings.monitoring_configs, 'w') as f:
                    json.dump(ms, f, indent=4) 
            print(f"Next check will be in {time_between_checks} seconds")
            await asyncio.sleep(time_between_checks)
        except FileNotFoundError:
            print("Error: The monitoring config file does not exist.")
            break

