import streamlit as st
import asyncio
import os
from together import AsyncTogether, Together

st.title("Mixture-of-Agents LLM App")

# API Key Setup
together_api_key = st.text_input("Enter your Together API Key:", type="password")
if together_api_key:
    os.environ["TOGETHER_API_KEY"] = together_api_key
    client = Together(api_key=together_api_key)
    async_client = AsyncTogether(api_key=together_api_key)

# Models that work reliably
reference_models = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3-70b-chat-hf"
]

aggregator_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
aggregator_system_prompt = """
You have been provided with responses from multiple AI models to the user's query. 
Your task is to synthesize these into a single, high-quality response by:

1. Comparing and contrasting the different responses
2. Identifying the most accurate and valuable information
3. Filtering out any incorrect or biased content
4. Creating a comprehensive, well-structured answer

Critically evaluate all information before including it. The final response should be:
- Accurate and factual
- Well-organized and coherent
- Comprehensive yet concise
- Directly addressing the user's original question

Model responses:
{responses}
"""

# Async function to run individual models
async def run_llm(model, user_prompt):
    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        return model, response.choices[0].message.content
    except Exception as e:
        return model, f"Error: {str(e)}"

# Main processing function
async def main(user_prompt):
    # Run all models concurrently
    tasks = [run_llm(model, user_prompt) for model in reference_models]
    results = await asyncio.gather(*tasks)
    
    # Display individual responses
    st.subheader("Individual Model Responses:")
    for model, response in results:
        with st.expander(f"Response from {model}"):
            st.write(response)
    
    # Prepare aggregated input
    response_texts = "\n\n".join([f"### {model}:\n{response}" for model, response in results])
    final_prompt = aggregator_system_prompt.format(responses=response_texts)
    
    # Generate aggregated response with streaming
    st.subheader("Aggregated Response:")
    response_container = st.empty()
    full_response = ""
    
    try:
        final_stream = client.chat.completions.create(
            model=aggregator_model,
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )
        
        # Stream aggregated response
        for chunk in final_stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                response_container.markdown(full_response + "▌")
        response_container.markdown(full_response)
    
    except Exception as e:
        st.error(f"Failed to generate aggregated response: {str(e)}")
        st.info("Troubleshooting tips:")
        st.write("1. Check your Together.ai account balance")
        st.write("2. Verify model availability at [Together Models](https://api.together.ai/models)")
        st.write("3. Try reducing max_tokens if responses are too long")
        st.write(f"Error details: {str(e)}")

# User input
user_prompt = st.text_area("Enter your question:", height=150)
if st.button("Get Answer"):
    if not together_api_key:
        st.error("Please enter your Together API Key")
    elif not user_prompt.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Querying AI models..."):
            asyncio.run(main(user_prompt))
