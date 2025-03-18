import openai  
from langchain.llms import OpenAI  
from langchain.chains import LLMChain  
from langchain.prompts import PromptTemplate  
from langchain.memory import ConversationBufferMemory  

# OpenAI API Key (replace with your actual key)
openai.api_key = "your_openai_api_key"

# Define Prompt Engineering Template
resume_prompt = PromptTemplate(
    input_variables=["resume_text"],
    template="Analyze the following resume and provide actionable improvement suggestions:\n\n{resume_text}"
)

# Initialize AI Model (OpenAI GPT)
llm = OpenAI(model_name="gpt-4")

# Create LangChain Memory (for interactive conversations)
memory = ConversationBufferMemory()

# Create AI Processing Chain
resume_chain = LLMChain(llm=llm, prompt=resume_prompt, memory=memory)

def analyze_resume(resume_text):
    """Processes resume text and returns AI-generated feedback."""
    return resume_chain.run(resume_text)
