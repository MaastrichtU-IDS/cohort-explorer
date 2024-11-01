from rag.llm_chain import evaluate_final_mapping
import re


def clean_output(explanation, predicted_class):
    # Remove extra newlines and unnecessary spaces
    explanation_clean = explanation.replace("\n", " ").strip()
    predicted_class_clean = predicted_class.strip()

    # Format output as a single string with clean explanation and prediction
    return explanation_clean, predicted_class_clean


EXPLANATION_PATTERN = re.compile(
    r"(.*)\s*\*\*final classification:\*\*\s*(.*)", re.IGNORECASE | re.DOTALL
)


# def perform_eval_for_variable(variable):
#     explanation = evaluate_final_mapping(variable, LLM_ID)
#     explanation = explanation.lower().strip()
#     print(f"Explanation for {variable}: {explanation}")
#     match = EXPLANATION_PATTERN.search(explanation)
#     if match:
#         explanation, prediction = match.groups()
#     else:
#         # If no match is found, use default values
#         explanation = explanation
#         prediction = "unknown"
#         if "correct" in explanation:
#             prediction = "correct"
#         elif "incorrect" in explanation:
#             prediction = "incorrect"
#         elif "partially correct" in explanation:
#             prediction = "partially correct"

#     # Now call clean_output with the explanation and prediction and use its result
#     cleaned_reasoning, cleaned_prediction = clean_output(explanation, prediction)
#     pred_dict = {
#         "reasoning": cleaned_reasoning,  # Cleaned reasoning as text
#         "prediction": cleaned_prediction,  # Cleaned prediction as text
#     }
#     print(pred_dict)
#     return pred_dict


def perform_mapping_eval_for_variable(var_: dict, llm_id: str = "llama3.1"):
    variable = var_
    # delete VARIABLE NAME key from variable
    # variable.pop("VARIABLE NAME")
    explanation = evaluate_final_mapping(variable, llm_id)
    explanation = explanation.lower().strip()
    print(f"Explanation for {variable['VARIABLE LABEL']}: {explanation}")
    match = EXPLANATION_PATTERN.search(explanation)
    if match:
        explanation, prediction = match.groups()
    else:
        # If no match is found, use default values
        explanation = explanation
        prediction = "unknown"

        if "incorrect" in explanation:
            prediction = "incorrect"
        elif "partially correct" in explanation:
            prediction = "partially correct"
        elif "partially incorrect" in explanation:
            prediction = "partially correct"
        elif "correct" in explanation:
            prediction = "correct"

    # Now call clean_output with the explanation and prediction and use its result
    cleaned_reasoning, cleaned_prediction = clean_output(explanation, prediction)
    variable["reasoning"] = cleaned_reasoning
    variable["prediction"] = cleaned_prediction
    # variable["VARIABLE NAME"] = var_["VARIABLE NAME"]
    return variable
