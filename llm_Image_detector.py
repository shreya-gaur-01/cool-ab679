# %%
import base64
import os
import asyncio
from typing import TypedDict, List, Literal, Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# %%
# --- 1. Define the Shared State ---
class RouterState(TypedDict):
    image_path: str                     # Path to local file (e.g., "data/car_and_apple.jpg")
    selected_models: List[str]          # Models selected by the LLM Router
    flower_detections: Optional[List]   
    fruit_detections: Optional[List]    
    vehicle_detections: Optional[List]  
    final_report: Optional[dict]       

# %%
class ModelSelection(BaseModel):
    selected_models: List[Literal["model_flowers", "model_fruits", "model_vehicles"]] = Field(
        description="Specialised computer vision models required based on what items exist in the image."
    )

# --- 2. Helper to Encode Local Image to Base64 ---
def encode_image_to_base64(image_path: str) -> str:
    """Reads a local image file and converts it into a Base64 Data URI."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
        
    # Determine the image format extension
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
        
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
    return f"data:image/{ext};base64,{encoded_string}"

# %%
# --- 3. The Router Node (Accepts Local Base64 String) ---
def image_router_node(state: RouterState) -> dict:
    print(f"[Router Node] Encoding and inspecting local file: {state['image_path']}...")
    
    # Convert the local path into a valid multimodal image URL format
    base64_image_url = encode_image_to_base64(state["image_path"])
    
    from langchain_groq import ChatGroq
    # 1. Initialize your LLM
    groq_api = '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'
    llm = ChatGroq(model="qwen/qwen3.6-27b", api_key=groq_api,
                max_tokens=900,
               temperature=0.1).with_structured_output(ModelSelection)
        
    message = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": "Analyze this image. Which of these classifiers do we need to activate? 'model_flowers' (if flowers present), 'model_fruits' (if fruits present), or 'model_vehicles' (if cars/bikes/vehicles present). Select all that apply."
            },
            {
                "type": "image_url", 
                "image_url": {"url": base64_image_url} # Passes encoded local image
            }
        ]
    )
    
    decision = llm.invoke([message])
    print(f"  --> LLM decided to activate: {decision.selected_models}")
    return {"selected_models": decision.selected_models}


# %%
 # --- 4. The Rest of the Graph Remains the Same ---
def predict_flowers(state: RouterState) -> dict:
    import tensorflow as tf
    model = tf.keras.models.load_model("model/FLOWER.keras")
    #Prediction
    img = tf.keras.utils.load_img(state['image_path'], target_size=(64,64))

    x = tf.keras.utils.img_to_array(img)
    x = tf.expand_dims(x, 0)

    p = model.predict(x)
    classes = ["daisy", "rose", "sunflower", "tulip"]
    cl=classes[p.argmax()]
    conf=p.max()
    print("[TF Node] Processing flowers model using local image file...",cl)
    # Your local tf_model.predict logic goes here using state["image_path"]
    return {"flower_detections": [{"label": cl, "confidence": conf}]}

def predict_fruits(state: RouterState) -> dict:
       import tensorflow as tf
       model = tf.keras.models.load_model("model/Fruits.keras")
       #Prediction
       img = tf.keras.utils.load_img(state['image_path'], target_size=(64,64))
   
       x = tf.keras.utils.img_to_array(img)
       x = tf.expand_dims(x, 0)
   
       p = model.predict(x)
       classes = ["apple", "banana", "mango", "orange"]
       cl=classes[p.argmax()]
       conf=p.max()
       print("[TF Node] Processing flowers model using local image file...",cl)
       
       print("[TF Node] Processing fruits model using local image file...")
       return {"fruit_detections": [{"label": cl, "confidence": conf}]}

def predict_vehicles(state: RouterState) -> dict:
    import tensorflow as tf
    model = tf.keras.models.load_model("model/vehicle.keras")
    #Prediction
    img = tf.keras.utils.load_img(state['image_path'], target_size=(64,64))
       
    x = tf.keras.utils.img_to_array(img)
    x = tf.expand_dims(x, 0)
       
    p = model.predict(x)
    classes = ["ambulance", "fire brigade", "police car", "rescue helicopter"]
    cl=classes[p.argmax()]
    conf=p.max()
              
    print("[TF Node] Processing vehicles model using local image file...")
    return {"vehicle_detections": [{"label": cl, "confidence": conf}]}

def route_to_models(state: RouterState) -> List[str]:
    if not state["selected_models"]:
        return ["aggregator"]
    return state["selected_models"]

def aggregate_results(state: RouterState) -> dict:
    print("\n[Aggregator] Compiling report...")
    report = {
        "flowers": state.get("flower_detections", "Not scanned"),
        "fruits": state.get("fruit_detections", "Not scanned"),
        "vehicles": state.get("vehicle_detections", "Not scanned")
    }
    return {"final_report": report}



# %%
# --- 5. Constructing Graph ---
workflow = StateGraph(RouterState)
workflow.add_node("llm_router", image_router_node)
workflow.add_node("model_flowers", predict_flowers)
workflow.add_node("model_fruits", predict_fruits)
workflow.add_node("model_vehicles", predict_vehicles)
workflow.add_node("aggregator", aggregate_results)

workflow.add_edge(START, "llm_router")
workflow.add_conditional_edges(
    "llm_router",
    route_to_models,
    path_map={
        "model_flowers": "model_flowers",
        "model_fruits": "model_fruits",
        "model_vehicles": "model_vehicles",
        "aggregator": "aggregator"
    }
)
workflow.add_edge("model_flowers", "aggregator")
workflow.add_edge("model_fruits", "aggregator")
workflow.add_edge("model_vehicles", "aggregator")
workflow.add_edge("aggregator", END)

app = workflow.compile()

# %%
# --- 6. Execution ---
def main():
    # Pass your actual local image file path here
    initial_payload = {
        "image_path": "do2.jpeg", # Ensure this file exists in your workspace
        "selected_models": []
    }
    
    try:
        result = app.invoke(initial_payload)
        print("\nFinal Dynamic Graph Result:")
        print(result["final_report"])
    except FileNotFoundError as e:
        print(f"Error: {e}. Please place a valid image in your path to test.")

def predict_image(img_file):
    # Pass your actual local image file path here
    initial_payload = {
        "image_path": img_file, # Ensure this file exists in your workspace
        "selected_models": []
    }
    
    try:
        result = app.invoke(initial_payload)
        #print("\nFinal Dynamic Graph Result:")
        #print(result["final_report"])
        return result["final_report"]
    except FileNotFoundError as e:
        print(f"Error: {e}. Please place a valid image in your path to test.")




