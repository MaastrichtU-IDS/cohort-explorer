import sqlite3
import csv
import io


def create_sqlite_database(filename):
    """create a database connection to an SQLite database"""
    conn = None
    try:
        conn = sqlite3.connect(filename, timeout=30.0)
        print(f"Connected to {filename}---{conn}")
    except sqlite3.Error as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()
            print(f"Connection closed to {filename}")


# create_sqlite_database("my.db")

# import os

# Sample data (as provided)
data = """VARIABLE NAME\tVARIABLE LABEL\tDomain\tvar\tVariable Concept Code\tVariable Concept OMOP ID\tAdditional Context Concept Label\tAdditional Context Concept Code\tAdditional Context OMOP ID\tPrimary to Secondary Context Relationship\tCategorical Values Concept Label\tCategorical Values Concept Code\tCategorical Values Concept OMOP ID\tUnit Concept Label\tUnit Concept Code\tUnit OMOP ID
month30_a\tfollow-up 30 months\tvisit\tfollow-up 30 months\ticare:icv2000000001\2000000001\tfollow-up 30 months\ticare:icv2000000001\2000000001\thas temporal context\t\t\t\t
month30_b\tfollow-up 30 month\tvisit\tfollow-up 30 months\ticare:icv2000000001\2000000001\tfollow-up 30 months\ticare:icv2000000001\2000000001\thas temporal context\t\t\t\t
month30_c\tfollow-up month 30\tvisit\tfollow-up 30 months\ticare:icv2000000001\2000000001\tfollow-up 30 months\ticare:icv2000000001\2000000001\thas temporal context\t\t\t\t
month30_d\t30 month follow up\tvisit\tfollow-up 30 months\ticare:icv2000000001\2000000001\tfollow-up 30 months\ticare:icv2000000001\2000000001\thas temporal context\t\t\t\t
month42_d\t42 month follow up\tvisit\tfollow-up 30 months\ticare:icv2000000001\2000000001\tfollow-up 30 months\ticare:icv2000000001\2000000001\thas temporal context\t\t\t\t

"""


class DataManager:
    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.create_table()

        # Strip

    # Specify the database file name

    # Create the 'variable' table if it doesn't exist
    def create_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS variable (
            variable_name TEXT PRIMARY KEY,
            variable_label TEXT UNIQUE,
            domain TEXT,
            variable_concept_label TEXT,
            variable_concept_code TEXT,
            variable_concept_omop_id INTEGER,
            additional_context_concept_label TEXT,
            additional_context_concept_code TEXT,
            additional_context_omop_id TEXT,
            primary_to_secondary_context_relationship TEXT,
            categorical_values_concept_label TEXT,
            categorical_values_concept_code TEXT,
            categorical_values_omop_id INTEGER,
            unit_concept_label TEXT,
            unit_concept_code TEXT,
            unit_omop_id INTEGER,
            reasoning TEXT,
            prediction TEXT
        );
        """
        cursor = self.conn.cursor()
        cursor.execute(create_table_sql)
        self.conn.commit()

    def variable_name_exists(self, variable_name):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM variable
            WHERE variable_name = ?
        """,
            (variable_name,),
        )
        results = cursor.fetchall()
        return bool(results)

    def insert_data_if_empty(self, data):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM variable")
        count = cursor.fetchone()[0]
        if count == 0:
            print("Inserting data into the 'variable' table...")
            reader = csv.DictReader(io.StringIO(data), delimiter="\t")
            headers = reader.fieldnames
            # Strip any leading/trailing whitespace from the headers
            reader.fieldnames = [header.strip() for header in headers]

            to_db = []
            for row in reader:
                # Prepare the data, handle missing values and type conversions
                to_db.append(
                    (
                        row["VARIABLE NAME"],
                        row["VARIABLE LABEL"],
                        row["Variable Concept Label"],
                        row["Variable Concept Code"],
                        str(row["Variable Concept OMOP ID"])
                        if row["Variable Concept OMOP ID"]
                        else None,
                        row["Domain"],
                        row["Additional Context Concept Label"],
                        row["Additional Context Concept Code"],
                        row["Additional Context OMOP ID"],
                        row["Primary to Secondary Context Relationship"],
                        row["Categorical Values Concept Label"],
                        row["Categorical Values Concept Code"],
                        str(row["Categorical Values Concept OMOP ID"])
                        if row["Categorical Values Concept OMOP ID"]
                        else None,
                        row["Unit Concept Label"],
                        row["Unit Concept Code"],
                        str(row["Unit Concept OMOP ID"])
                        if row["Unit Concept OMOP ID"]
                        else None,
                        str(row["Reasoning"]) if row["Reasoning"] else None,
                        str(row["Prediction"]) if row["Prediction"] else None,
                    )
                )
            # Insert data into the table
            cursor.executemany(
                """
                INSERT INTO variable (
                    variable_name,
                    variable_label,
                    variable_concept_label,
                    variable_concept_code,
                    variable_concept_omop_id,
                    domain,
                    additional_context_concept_label,
                    additional_context_concept_code,
                    additional_context_omop_id,
                    primary_to_secondary_context_relationship,
                    categorical_values_concept_label,
                    categorical_values_concept_code,
                    categorical_values_omop_id,
                    unit_concept_label,
                    unit_concept_code,
                    unit_omop_id,
                    reasoning,
                    prediction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                to_db,
            )
            self.conn.commit()
        else:
            print("Data already exists in the 'variable' table.")

    def dict_keys_to_columns(self, row_data):
        return {k.lower().replace(" ", "_"): v for k, v in row_data.items()}

    def insert_row(self, row_data) -> dict:
        if row_data is None:
            return {
                "status": "error",
                "message": "No data provided for insertion.",
            }

        # Convert dict keys to match database column format
        row_data = self.dict_keys_to_columns(row_data)

        # Extract variable_name and variable_label from row_data
        variable_name = row_data.get("variable_name", None)
        variable_label = row_data.get("variable_label", None)

        # Ensure both variable_name and variable_label are provided
        if not variable_name or not variable_label:
            return {
                "status": "error",
                "message": "Both 'variable_name' and 'variable_label' are required.",
            }

        # Check if a record with the same variable_name and variable_label already exists
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT 1 FROM variable
            WHERE variable_name = ? AND variable_label = ?
            """,
            (variable_name, variable_label),
        )

        # If the query returns a result, that means a record with the same variable_name and variable_label exists
        if cursor.fetchone():
            print(
                f"Row with variable_name '{variable_name}' and variable_label '{variable_label}' already exists."
            )
            return {
                "status": "info",
                "message": f"IntegrityConstraint: Record with variable_name '{variable_name}' and variable_label '{variable_label}' already exists.",
            }

        # Proceed with the insertion if no duplicate is found
        columns = [
            "variable_name",
            "variable_label",
            "domain",
            "variable_concept_label",
            "variable_concept_code",
            "variable_concept_omop_id",
            "additional_context_concept_label",
            "additional_context_concept_code",
            "additional_context_omop_id",
            "primary_to_secondary_context_relationship",
            "categorical_values_concept_label",
            "categorical_values_concept_code",
            "categorical_values_omop_id",
            "unit_concept_label",
            "unit_concept_code",
            "unit_omop_id",
            "reasoning",
            "prediction",
        ]

        try:
            # Ensure that all required keys are present in the input dictionary
            missing_columns = [col for col in columns if col not in row_data]
            if missing_columns:
                return {
                    "status": "error",
                    "message": f"Missing columns in input data: {missing_columns}",
                }

            # Extract values in the correct order
            values = [row_data.get(col) for col in columns]

            # Construct the SQL insert statement
            placeholders = ", ".join("?" * len(columns))
            sql = f"""
                INSERT INTO variable ({', '.join(columns)})
                VALUES ({placeholders})
            """
            cursor.execute(sql, values)
            self.conn.commit()
            print(
                f"Row inserted successfully with variable_name '{variable_name}' and variable_label '{variable_label}'."
            )
            return {
                "status": "success",
                "message": f"Row inserted successfully with variable_name '{variable_name}' and variable_label '{variable_label}'.",
            }

        except sqlite3.IntegrityError as e:
            # Handle unique constraint failure
            return {"status": "error", "message": f"IntegrityError: {e}"}
        except Exception as e:
            # Handle any other exceptions
            return {"status": "error", "message": f"An error occurred: {e}"}

    # Function to perform the query based on the given string

    def select_all(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM variable")
        rows = cursor.fetchall()
        return rows

    # def extract_all(self):
    #     cursor = self.conn.cursor()
    #     cursor.execute("SELECT * FROM variable")
    #     rows = cursor.fetchall()
    #     return rows

    def query_variable_name(self, input_string, ):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM variable
            WHERE LOWER(variable_name) = LOWER(?)
        """,
            (input_string,),
        )
        results = cursor.fetchall()
        if results:
            # Return all columns for the matching row(s)
            # for row in results:
            #     print(row)
            return results, "full"
        else:
            print("Input string not found.")
            return None, ""

    def query_variable(self, input_string, var_name=None):
        input_string = input_string.strip().lower()
        print(f"string to search for {input_string}")

        cursor = self.conn.cursor()
        # step 1.3 check if input_String exists in variable label and variable name (if provided)
        if var_name:
            # check for both variable name and variable label
            cursor.execute(
                """
                SELECT *
                FROM variable
                WHERE LOWER(variable_name) = LOWER(?) and LOWER(variable_label) = LOWER(?)
            """,
        
                (var_name, input_string),
            )
        else:
            cursor.execute(
                """
                    SELECT *
                    FROM variable
                    WHERE LOWER(variable_label) = LOWER(?)
                """,
                (input_string,),
            )
        results = cursor.fetchall()
        print(f"result for search in label {results}")
        if results:
            # Return all columns for the matching row(s)
            print("Found in variable label:")
            # for row in results:
            #     print(row)
            found = True
            return results[0], "full"

        # Step 2: Check if input_string exists in categorical_values_concept_label
        cursor.execute(
            """
            SELECT
                variable_name,
                categorical_values_concept_label,
                categorical_values_concept_code,
                categorical_values_omop_id
            FROM variable
            WHERE LOWER(categorical_values_concept_label) = LOWER(?)
        """,
            ("%" + input_string + "%",),
        )
        results = cursor.fetchall()
        if results:
            # Return categorical label, code, omop_id
            print("Found in categorical_values_concept_label:")
            # for row in results:
            #     print(row)
            found = True
            return results[0], "subset"
        # step 2.1 check if input_String exists in variable_concept_label
        cursor.execute(
            """
            SELECT variable_name, variable_concept_label, variable_concept_code, variable_concept_omop_id
            FROM variable
            WHERE LOWER(variable_concept_label) = LOWER(?)
            """,
            ("%" + input_string + "%",),
        )
        result = cursor.fetchall()
        if result:
            print("Found in variable_concept_label:")
            # for row in result:
            #     print(row)
            found = True
            return result[0], "subset"
        # Step 3: Check if input_string exists in additional_context_concept_label
        # which can be pipe-separated
        # We'll need to split the values and check for matches
        cursor.execute("""
            SELECT
                variable_name,
                additional_context_concept_label,
                additional_context_concept_code,
                additional_context_omop_id
            FROM variable
            WHERE additional_context_concept_label IS NOT NULL
        """)
        rows = cursor.fetchall()
        found = False
        for row in rows:
            variable_name, labels, codes, omop_ids = row
            label_list = labels.split("|")
            code_list = codes.split("|") if codes else []
            omop_id_list = omop_ids.split("|") if omop_ids else []
            if input_string in label_list:
                index = label_list.index(input_string)
                code = code_list[index] if index < len(code_list) else None
                omop_id = omop_id_list[index] if index < len(omop_id_list) else None
                # print("Found in additional_context_concept_label:")
                # print("Variable Name:", variable_name)
                # print("Label:", input_string)
                # print("Code:", code)
                # print("OMOP ID:", omop_id)
                found = True
                return (variable_name, input_string, code, omop_id), "subset"
        # step check the string in Unit concept label
        cursor.execute(
            """
            SELECT variable_name, unit_concept_label, unit_concept_code, unit_omop_id
            FROM variable
            WHERE LOWER(unit_concept_label) = LOWER(?)
            """,
            ("%" + input_string + "%",),
        )
        result = cursor.fetchall()
        if result:
            print("Found in unit_concept_label:")
            # for row in result:
            #     print(row)
            found = True
            return result[0], "subset"
        if not found:
            print("Input string not found.")
        return None, ""

    # Function to handle main term and associated entities
    def query_relationship(self, main_term, associated_entities):
        cursor = self.conn.cursor()
        # Step 1: Check if main_term exists in variable_label
        cursor.execute(
            """
            SELECT *
            FROM variable
            WHERE variable_concept_label = ?
        """,
            (main_term,),
        )
        variables = cursor.fetchall()
        if not variables:
            print(f"Main term '{main_term}' not found in variable_label.")
            return None

        # For each matching variable, check associated_entities in additional_context_concept_label
        for variable in variables:
            variable_name = variable[0]
            additional_context_labels = variable[
                6
            ]  # Index of additional_context_concept_label
            relationship = variable[
                9
            ]  # Index of primary_to_secondary_context_relationship
            if additional_context_labels:
                label_list = additional_context_labels.split("|")
                matching_entities = [
                    entity for entity in associated_entities if entity in label_list
                ]
                if matching_entities:
                    return (variable_name, main_term, matching_entities, relationship)
        return None

    def close_connection(self):
        self.conn.close()

db = DataManager("variables.db")
try:
    with open(
        "../data/custom_codes.csv",
        "r",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        for row in reader:
            result = db.insert_row(row)
            print(result)
except Exception as e:
    print(e)
# print(db.query_variable("c-reactive protein high sensitivity at baseline time"))
# # Create the table if it doesn't exist
# # db.create_table()

# # # # # Insert data if the table is empty
# # db.insert_data_if_empty(data)


# # # Example usage
# print("=== Query for input string 'heart rate' ===")
# print(db.query_variable('heart rate'))


# print("=== Query for input string 'diastolic blood pressure' ===")
# print(db.query_variable('systolic blood pressure'))

# print("\n=== Query for input string 'missing' ===")
# print(db.query_variable('missing'))

# print("\n=== Query for input string 'standing' ===")
# print(db.query_variable('standing'))

# print(db.query_relationship('heart rate', ['standing', 'follow-up 1 month']))

# print(db.insert_row({'VARIABLE NAME': 'bpdiast12', 'VARIABLE LABEL': 'bp blood pressure diastolic in recumbent position at visit month 12 month 12',
#                   'Domain': 'measurement', 'Variable Concept Label': 'diastolic blood pressure', 'Variable Concept Code': 'loinc:8462-4', 'Variable Concept OMOP ID': '3012888',
#                   'Additional Context Concept Label': 'recumbent body position|follow-up 2 months', 'Additional Context Concept Code': 'snomed:102538003|snomed:200891000000107', 'Additional Context OMOP ID': '4009887|44788749', 'Primary to Secondary Context Relationship': 'has temporal context',
#                   'Categorical Values Concept Label': 'missing',
#                   'Categorical Values Concept Code': 'loinc:LA14698-7',
#                   'Categorical Values Concept OMOP ID': '45882933', 'Unit Concept Label': 'millimeter mercury column',
#                   'Unit Concept Code': 'ucum:mm[Hg]', 'Unit OMOP ID': '8876'}))


# # Close the database connection
# CONN.close()
# db = DataManager("variables.db")
# import pandas as pd

# unmapped_rows = []
# df = pd.read_csv(
#     "/workspace/mapping_tool/data/eval_datasets/Icare_data/iCARE4CVD_GISSI_HF.csv"
# )
# for index, row in df.iterrows():
#     variable_name = row["VARIABLE NAME"]
#     if pd.notna(variable_name):
#         result, _ = db.query_variable_name(variable_name.strip().lower())
#         if not result or len(result) == 0:
#             unmapped_rows.append(row)
#         else:
#             print("found")

# unmapped_df = pd.DataFrame(unmapped_rows)
# if not unmapped_df.empty:
#     unmapped_df.columns = df.columns  # Assign columns only if DataFrame is not empty

# # Save to CSV if there are unmapped rows
# if not unmapped_df.empty:
#     unmapped_df.to_csv(
#         "/workspace/mapping_tool/data/eval_datasets/Icare_data/GISSI_HF_part2.csv",
#         index=False,
#     )
#     print(f"Length of unmapped rows: {len(unmapped_rows)}")
# else:
#     print("No unmapped rows found.")
