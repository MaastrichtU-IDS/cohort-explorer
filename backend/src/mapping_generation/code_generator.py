class CodeGenerator:
    def __init__(self, start_value=2000000000):
        # Initialize the starting value above 2 billion to comply with OMOP standards.
        self.counters = {
            'COND': start_value,
            'ANAT_SITE': start_value,
            'BODY_STRUCT': start_value,
            'MEAS': start_value,
            'PROC': start_value,
            'MED': start_value,
            'DEV': start_value,
            'UNIT': start_value,
            'VISIT': start_value,
            'DEATH': start_value,
            'DEM': start_value,
            'FAM_HIST': start_value,
            'LIFE_STYLE': start_value,
            'HIST_EVENTS': start_value
        }

    def generate_code(self, domain, suffix=None):
        # Increment the counter for the given domain and return the unique code.
        if domain not in self.counters:
            raise ValueError(f"Invalid domain: {domain}")
        
        self.counters[domain] += 1
        base_code = f"ICV_{domain}_{self.counters[domain]}"
        return f"{base_code}_{suffix}" if suffix else base_code

# Example usage
code_gen = CodeGenerator()
print(code_gen.generate_code('COND'))  # Output: ICV_COND_2000000001
print(code_gen.generate_code('MED'))   # Output: ICV_MED_2000000001
print(code_gen.generate_code('COND'))  # Output: ICV_COND_2000000002
print(code_gen.generate_code('MEAS'))  # Output: ICV_MEAS_2000000001

# Generating codes with specific months or years suffixes
print(code_gen.generate_code('VISIT', '30M'))  # Output: ICV_VISIT_2000000001_30M
print(code_gen.generate_code('VISIT', '42M'))  # Output: ICV_VISIT_2000000002_42M
print(code_gen.generate_code('VISIT', '48M'))  # Output: ICV_VISIT_2000000003_48M
print(code_gen.generate_code('VISIT', '3Y'))   # Output: ICV_VISIT_2000000004_3Y
print(code_gen.generate_code('VISIT', '54M'))  # Output: ICV_VISIT_2000000005_54M
print(code_gen.generate_code('VISIT', '60M'))  # Output: ICV_VISIT_2000000006_60M

# Example usage
# code_gen = CodeGenerator()
# print(code_gen.generate_code('ICV_COND'))  # Output: ICV_COND_2000000001
# print(code_gen.generate_code('ICV_MED'))   # Output: ICV_MED_2000000001
# print(code_gen.generate_code('ICV_COND'))  # Output: ICV_COND_2000000002
# print(code_gen.generate_code('ICV_MEAS'))  # Output: ICV_MEAS_2000000001
