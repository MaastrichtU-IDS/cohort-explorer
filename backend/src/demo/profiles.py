"""Stable semantic-to-real-column profiles for the two synthetic cohorts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariableBinding:
    semantic: str
    visit: str
    source: str
    target: str


@dataclass(frozen=True)
class CohortProfile:
    cohort_id: str
    side: str
    bindings: tuple[VariableBinding, ...]

    def column(self, semantic: str, visit: str = "baseline") -> str:
        for binding in self.bindings:
            if binding.semantic == semantic and binding.visit == visit:
                return binding.source if self.side == "source" else binding.target
        raise KeyError(f"No {semantic!r} variable at {visit!r} for {self.cohort_id}")

    @property
    def columns(self) -> tuple[str, ...]:
        attribute = "source" if self.side == "source" else "target"
        return tuple(getattr(binding, attribute) for binding in self.bindings)


SELECTED_BINDINGS = (
    VariableBinding("patient_id", "baseline", "patientid", "codr1"),
    VariableBinding("age", "baseline", "age", "age"),
    VariableBinding("sex", "baseline", "gender", "sesso"),
    VariableBinding("diabetes", "baseline", "diabetes", "prediab"),
    VariableBinding("hypertension", "baseline", "hypertension", "preiper"),
    VariableBinding("smoking", "baseline", "smoking", "prefumo"),
    VariableBinding("systolic_pressure", "baseline", "bpsyst", "pas"),
    VariableBinding("systolic_pressure", "3m", "bpsyst3", "f4_pas"),
    VariableBinding("systolic_pressure", "1y", "bpsyst12", "f6_pas"),
    VariableBinding("diastolic_pressure", "baseline", "bpdiast", "pad"),
    VariableBinding("diastolic_pressure", "3m", "bpdiast3", "f4_pad"),
    VariableBinding("diastolic_pressure", "1y", "bpdiast12", "f6_pad"),
    VariableBinding("heart_rate", "baseline", "hr", "fc"),
    VariableBinding("weight", "baseline", "weight", "pesoatt"),
    VariableBinding("weight", "3m", "weight3", "f4_pesoatt"),
    VariableBinding("weight", "1y", "weight12", "f6_pesoatt"),
    VariableBinding("height", "baseline", "height", "altezza"),
    VariableBinding("ejection_fraction", "baseline", "ef", "valfe"),
    VariableBinding("nt_pro_bnp", "baseline", "nbnp", "v1_nt_probnp"),
    VariableBinding("creatinine", "baseline", "creatinine", "creatin"),
    VariableBinding("creatinine", "3m", "crea3", "f4_creatin"),
    VariableBinding("creatinine", "1y", "crea12", "f6_creatin"),
    VariableBinding("hemoglobin", "baseline", "hb", "emoglob"),
    VariableBinding("nyha_class", "baseline", "nyha_class", "nyha"),
    VariableBinding("nyha_class", "3m", "nyha3_class", "f4_nyha"),
    VariableBinding("nyha_class", "1y", "nyha12_class", "f6_nyha"),
    VariableBinding("furosemide_exposed", "baseline", "blfuro", "furosemi"),
    VariableBinding("furosemide_dose", "baseline", "dosefurobl", "dfurosem"),
    VariableBinding("furosemide_exposed", "3m", "v3furo", "f4_furosemi"),
    VariableBinding("furosemide_dose", "3m", "dosefurov3", "f4_dfurosem"),
    VariableBinding("furosemide_exposed", "1y", "v12furo", "f6_furosemi"),
    VariableBinding("furosemide_dose", "1y", "dosefurov12", "f6_dfurosem"),
    VariableBinding("spironolactone_exposed", "baseline", "spiroadjbl", "spirono"),
    VariableBinding("spironolactone_dose", "baseline", "dosespirobl", "dspirono"),
    VariableBinding("heart_failure_hospitalization", "baseline", "hf_hosp", "npreospe"),
)


COHORT_PROFILES = {
    "TIME-CHF": CohortProfile("TIME-CHF", "source", SELECTED_BINDINGS),
    "GISSI-HF": CohortProfile("GISSI-HF", "target", SELECTED_BINDINGS),
}


CATEGORICAL_ENCODINGS = {
    "sex": "0=Female | 1=Male",
    "diabetes": "0=No | 1=Yes",
    "hypertension": "0=No | 1=Yes",
    "smoking": "0=Never | 1=Former | 2=Current",
    "nyha_class": "1=NYHA I | 2=NYHA II | 3=NYHA III | 4=NYHA IV",
    "furosemide_exposed": "0=No | 1=Yes",
    "spironolactone_exposed": "0=No | 1=Yes",
    "heart_failure_hospitalization": "0=No | 1=Yes",
}
