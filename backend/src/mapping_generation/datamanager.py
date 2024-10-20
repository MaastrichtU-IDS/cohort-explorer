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
BPsyststand1\tBlood pressure systolic standing at visit month 1\tmeasurement\tsystolic blood pressure\tloinc:8480-6\t3004249\tstanding|follow-up 1 month\tloinc:LA11870-5|snomed:183623000\t45876596|4081745\thas temporal context\tmissing\tloinc:LA14698-7\t45882933\tmillimeter mercury column\tucum:mm[Hg]\t8876
BPdiaststand1\tBP blood pressure diastolic standing at visit month 1\tmeasurement\tdiastolic blood pressure\tloinc:8462-4\t3012888\tstanding|follow-up 1 month\tloinc:LA11870-5|snomed:183623000\t45876596|4081745\thas temporal context\tmissing\tloinc:LA14698-7\t45882933\tmillimeter mercury column\tucum:mm[Hg]\t8876
HRstand1\tHeart rate standing at visit month 1\tmeasurement\theart rate\tloinc:8867-4\t3027018\tstanding|follow-up 1 month\tloinc:LA11870-5|snomed:183623000\t45876596|4081745\thas temporal context\tmissing\tloinc:LA14698-7\t45882933\tcounts per minute\tucum:{counts}/min\t8483
BPsyststand3\tBP blood pressure systolic standing at visit month 3\tmeasurement\tsystolic blood pressure\tloinc:8480-6\t3004249\tstanding|follow-up 3 months\tloinc:LA11870-5|snomed:200521000000107\t45876596|44789369\thas temporal context\tmissing\tloinc:LA14698-7\t45882933\tmillimeter mercury column\tucum:mm[Hg]\t8876
BPdiaststand3\tBP blood pressure diastolic standing at visit month 3\tmeasurement\tdiastolic blood pressure\tloinc:8462-4\t3012888\tstanding|follow-up 3 months\tloinc:LA11870-5|snomed:200521000000107\t45876596|44789369\thas temporal context\tmissing\tloinc:LA14698-7\t45882933\tmillimeter mercury column\tucum:mm[Hg]\t8876
HRstand3\tHeart rate standing at visit month 3\tmeasurement\theart rate\tloinc:8867-4\t3027018\tstanding|follow-up 3 months\tloinc:LA11870-5|snomed:200521000000107\t45876596|44789369\thas temporal context\tmissing\tloinc:LA14698-7\t45882933\tcounts per minute\tucum:{counts}/min\t8483"""


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
            variable_label TEXT,
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
            categorical_values_concept_omop_id INTEGER,
            unit_concept_label TEXT,
            unit_concept_code TEXT,
            unit_omop_id INTEGER
        );
        """
        cursor = self.conn.cursor()
        cursor.execute(create_table_sql)
        self.conn.commit()

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
                        int(row["Variable Concept OMOP ID"])
                        if row["Variable Concept OMOP ID"]
                        else None,
                        row["Domain"],
                        row["Additional Context Concept Label"],
                        row["Additional Context Concept Code"],
                        row["Additional Context OMOP ID"],
                        row["Primary to Secondary Context Relationship"],
                        row["Categorical Values Concept Label"],
                        row["Categorical Values Concept Code"],
                        int(row["Categorical Values Concept OMOP ID"])
                        if row["Categorical Values Concept OMOP ID"]
                        else None,
                        row["Unit Concept Label"],
                        row["Unit Concept Code"],
                        int(row["Unit OMOP ID"]) if row["Unit OMOP ID"] else None,
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
                    categorical_values_concept_omop_id,
                    unit_concept_label,
                    unit_concept_code,
                    unit_omop_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        row_data = self.dict_keys_to_columns(row_data)
        """
        Inserts a single row into the 'variable' table.

        Parameters:
            conn (sqlite3.Connection): The database connection.
            row_data (dict): A dictionary containing the row data.

        Returns:
            dict: A standardized response indicating success or error.
        """
        results, mode = self.query_variable(row_data["variable_name"])
        if results:
            print(
                f"Row with variable_name '{row_data['variable_name']}' already exists."
                f" Results: {results}"
            )
            return {
                "status": "success",
                "message": f"Row with variable_name '{row_data['variable_name']}' already exists.",
            }
        cursor = self.conn.cursor()
        # List of all columns in the 'variable' table
        columns = [
            "variable_name",
            "variable_label",
            "variable_concept_label",
            "variable_concept_code",
            "variable_concept_omop_id",
            "domain",
            "additional_context_concept_label",
            "additional_context_concept_code",
            "additional_context_omop_id",
            "primary_to_secondary_context_relationship",
            "categorical_values_concept_label",
            "categorical_values_concept_code",
            "categorical_values_concept_omop_id",
            "unit_concept_label",
            "unit_concept_code",
            "unit_omop_id",
        ]

        # Prepare the data for insertion
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

            # Convert data types as necessary
            # Handle integer fields
            int_fields = [
                "variable_concept_omop_id",
                "categorical_values_concept_omop_id",
                "unit_omop_id",
            ]
            for idx, col in enumerate(columns):
                if col in int_fields and values[idx] is not None:
                    values[idx] = int(values[idx]) if values[idx] != "" else None

            # Construct the SQL insert statement
            placeholders = ", ".join("?" * len(columns))
            sql = f"""
                INSERT INTO variable ({', '.join(columns)})
                VALUES ({placeholders})
            """
            cursor.execute(sql, values)
            self.conn.commit()
            print(
                f"Row inserted successfully with variable_name '{row_data['variable_name']}'."
            )
            return {
                "status": "success",
                "message": f"Row inserted successfully with variable_name '{row_data['variable_name']}'.",
            }

        except sqlite3.IntegrityError as e:
            # Handle duplicate primary key error
            return {"status": "error", "message": f"IntegrityError: {e}"}
        except Exception as e:
            # Handle any other exceptions
            return {"status": "error", "message": f"An error occurred: {e}"}


    # Function to perform the query based on the given string
    def query_variable(self, input_string):
        print(f"string to search for {input_string}")
        cursor = self.conn.cursor()
        # Step 1: Check if input_string exists in variable_label
        cursor.execute(
            """
            SELECT *
            FROM variable
            WHERE variable_name = ?
        """,
            (input_string,),
        )
        results = cursor.fetchall()
        if results:
            # Return all columns for the matching row(s)
            print("Found in variable name")
            for row in results:
                print(row)
            found = True
            return results[0], "full"

        # step 1.3 check if input_String exists in variable label
        cursor.execute(
            """
            SELECT *
            FROM variable
            WHERE variable_label LIKE ?
        """,
            ("%" + input_string + "%",),
        )
        results = cursor.fetchall()
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
                categorical_values_concept_omop_id
            FROM variable
            WHERE categorical_values_concept_label LIKE ?
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
            WHERE variable_concept_label LIKE ?
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
            WHERE unit_concept_label LIKE ?
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


# db = DataManager('variables.db')

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
