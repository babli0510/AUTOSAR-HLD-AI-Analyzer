
import pymupdf

PDF_PATH = "data/raw/sample_hld.pdf"
PAGE_INDEX = 1

EXPECTED_ROWS = [
    [
        "Component",
        "Responsibility",
        "Main Inputs",
        "Main Outputs",
    ],
    [
        "SensorManager",
        "Acquire and validate temperature sensor values",
        "CabinTempRaw, AmbientTempRaw",
        "CabinTemperature, AmbientTemperature",
    ],
    [
        "TemperatureManager",
        "Calculate filtered temperatures and target deviation",
        "CabinTemperature, TargetTemperature",
        "TemperatureDeviation",
    ],
    [
        "ClimateController",
        "Determine climate operating mode and blower request",
        "TemperatureDeviation, VehicleState",
        "BlowerRequest, ClimateMode",
    ],
    [
        "FanController",
        "Convert blower request into actuator command",
        "BlowerRequest",
        "BlowerSpeedCommand",
    ],
    [
        "VehicleStateManager",
        "Provide vehicle operating state",
        "IgnitionState, VehicleSpeed",
        "VehicleState",
    ],
    [
        "DiagnosticManager",
        "Monitor faults and expose diagnostic status",
        "SensorStatus, ControllerStatus",
        "DiagnosticStatus",
    ],
    [
        "CommunicationManager",
        "Exchange climate signals with other ECUs",
        "VehicleState, ClimateStatus",
        "ClimateStatus",
    ],
]


def main():
    with pymupdf.open(PDF_PATH) as document:
        page = document[PAGE_INDEX]
        tables = page.find_tables()

        if not tables.tables:
            print("FAIL: No table found.")
            return

        detected_rows = tables.tables[0].extract()
        print("Detected rows:", len(detected_rows))
        print("Expected rows:", len(EXPECTED_ROWS))

        print("\nExpected table reconstructed for validation:")
        for index, row in enumerate(EXPECTED_ROWS, start=1):
            print(f"\nRow {index}:")
            for column, value in zip(
                ["Component", "Responsibility", "Main Inputs", "Main Outputs"],
                row,
            ):
                print(f"  {column}: {value}")

        print("\nRaw extraction still needs independent correction.")
        print("This test uses known sample content; it does not prove")
        print("that arbitrary uploaded PDFs can be reconstructed.")


if __name__ == "__main__":
    main()
