from rag.llm_chain import evaluate_final_mapping
import pandas as pd
import re
import time


def clean_output(explanation, predicted_class):
    # Remove extra newlines and unnecessary spaces
    explanation_clean = explanation.replace("\n", " ").strip()
    predicted_class_clean = predicted_class.strip()

    # Format output as a single string with clean explanation and prediction
    return f"Explanation: {explanation_clean}", predicted_class_clean


LLM_ID = "local_llama3.1"
EXPLANATION_PATTERN = re.compile(
    r"(.*)\s*\*\*final classification:\*\*\s*(.*)", re.IGNORECASE | re.DOTALL
)

def perform_eval_for_variable(variable):
    explanation = evaluate_final_mapping(variable, LLM_ID)
    explanation = explanation.lower().strip()
    print(f"Explanation for {variable}: {explanation}")
    match = EXPLANATION_PATTERN.search(explanation)
    if match:
        explanation, prediction = match.groups()
    else:
        # If no match is found, use default values
        explanation = explanation
        prediction = "unknown"
        if "correct" in explanation:
            prediction = "correct"
        elif "incorrect" in explanation:
            prediction = "incorrect"
        elif "partially correct" in explanation:
            prediction = "partially correct"

    # Now call clean_output with the explanation and prediction and use its result
    cleaned_reasoning, cleaned_prediction = clean_output(explanation, prediction)
    pred_dict = {
        "reasoning": cleaned_reasoning,  # Cleaned reasoning as text
        "prediction": cleaned_prediction,  # Cleaned prediction as text
    }
    print(pred_dict)
    return pred_dict
def perform_mapping_eval(csv_file):
    start_time = time.time()
    variables = pd.read_csv(csv_file)
    list_of_dictionaries = variables.to_dict(orient="records")
    print(f"length of list of dictionaries: {len(list_of_dictionaries)}")

    predictions = []
    for var in list_of_dictionaries:
        explanation = evaluate_final_mapping(var, LLM_ID)
        explanation = explanation.lower().strip()
        print(f"Explanation for {var['VARIABLE NAME']}: {explanation}")
        match = EXPLANATION_PATTERN.search(explanation)
        if match:
            explanation, prediction = match.groups()
        else:
            # If no match is found, use default values
            explanation = explanation
            prediction = "unknown"
            if "correct" in explanation:
                prediction = "correct"
            elif "incorrect" in explanation:
                prediction = "incorrect"
            elif "partially correct" in explanation:
                prediction = "partially correct"

        # Now call clean_output with the explanation and prediction and use its result
        cleaned_reasoning, cleaned_prediction = clean_output(explanation, prediction)

        pred_dict = {
            "variable": var["VARIABLE NAME"],
            "reasoning": cleaned_reasoning,  # Cleaned reasoning as text
            "prediction": cleaned_prediction,  # Cleaned prediction as text
        }
        print(pred_dict)
        predictions.append(pred_dict)

    # create csv file with reasoning and predictions
    df = pd.DataFrame(predictions)
    df.to_csv(
        f"/workspace/mapping_tool/data/output/reasoning_predictions_{LLM_ID}.csv",
        index=False,
    )
    print(f"Time taken: {time.time() - start_time}")


if __name__ == "__main__":
    perform_mapping_eval(
        "file_name.csv"
    )
    # perform_eval_for_variable("dict object") from retriever.py map_data function
    # we can add extra two columns for mapping confidence and reasoning -- correct, partially correct, incorrect
