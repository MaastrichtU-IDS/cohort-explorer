from pathlib import Path


class CohortVarLinkerConceptValidationProvider:
    def validate(self, dictionary_path: Path, report_path: Path) -> bool:
        from CohortVarLinker.validate_cde import validate_dictionary

        return bool(validate_dictionary(str(dictionary_path), str(report_path)))
