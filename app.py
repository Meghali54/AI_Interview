import streamlit as st
import json
import os
from PyPDF2 import PdfReader
from dataclasses import dataclass
from typing import Literal
import google.generativeai as genai

# Page config
st.set_page_config(
    page_title="AI Interview Question Generator - Gemini Powered", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<style>#MainMenu{visibility:hidden;}</style>",
    unsafe_allow_html=True
)

@dataclass
class InterviewQuestion:
    """Class for storing generated interview questions."""
    category: str
    question: str
    difficulty: Literal["Easy", "Medium", "Hard"]
    focus_area: str

def get_gemini_api_key():
    """Get Gemini API key from secrets or user input."""
    try:
        # Try to get from Streamlit secrets (for deployment)
        return st.secrets["gemini"]["api_key"]
    except:
        # Fallback to hardcoded key for local development
        return "AIzaSyCufEYr4Bf367qM-ADrOi-roMurHcVAHKk"

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF resume."""
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def generate_questions_with_gemini(resume_text, job_description, skills, num_questions, api_key):
    """Generate interview questions using Google Gemini API."""
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
You are an expert technical interviewer. Based on the provided resume, job description, and required skills, generate {num_questions} highly relevant interview questions.

RESUME CONTENT:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_description}

REQUIRED SKILLS:
{skills}

Generate exactly {num_questions} interview questions. For each question, format it as:
[Category|Difficulty|Focus Area] Question text

Where:
- Category: Technical, Behavioral, or Experience
- Difficulty: Easy, Medium, or Hard
- Focus Area: The specific skill or area being tested

Examples:
[Technical|Medium|Python Programming] Can you explain the difference between list comprehensions and generator expressions in Python?
[Behavioral|Easy|Team Collaboration] Tell me about a time when you had to work with a difficult team member.
[Experience|Hard|Problem Solving] Describe the most challenging technical problem you've solved and your approach.

Requirements for the questions:
1. Make them specific to the candidate's background from their resume
2. Ensure they test skills mentioned in the job description
3. Include a mix of technical and behavioral questions
4. Vary the difficulty levels appropriately
5. Make them practical and relevant to the role

Generate {num_questions} questions now in the specified format:
"""
        
        response = model.generate_content(prompt)
        return parse_gemini_questions(response.text, num_questions)
        
    except Exception as e:
        st.error(f"Error generating questions with Gemini: {str(e)}")
        return []

def parse_gemini_questions(response_text, num_questions):
    """Parse questions from Gemini response."""
    questions = []
    lines = response_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        try:
            # Look for pattern: [Category|Difficulty|Focus] Question
            if '[' in line and ']' in line and '|' in line:
                bracket_start = line.find('[')
                bracket_end = line.find(']')
                
                if bracket_start != -1 and bracket_end != -1:
                    bracket_content = line[bracket_start+1:bracket_end]
                    question_text = line[bracket_end+1:].strip()
                    
                    parts = [part.strip() for part in bracket_content.split('|')]
                    if len(parts) >= 3 and question_text:
                        category = parts[0]
                        difficulty = parts[1]
                        focus_area = parts[2]
                        
                        questions.append(InterviewQuestion(
                            category=category,
                            question=question_text,
                            difficulty=difficulty,
                            focus_area=focus_area
                        ))
        except Exception as e:
            # Skip malformed lines
            continue
    
    # If parsing failed, create fallback questions
    if len(questions) < 3:
        st.warning("Using fallback questions due to parsing issues")
        fallback_questions = [
            InterviewQuestion("Technical", "Tell me about your technical background and the technologies you're most comfortable with.", "Easy", "General Technical"),
            InterviewQuestion("Experience", "Walk me through a challenging project you've worked on recently.", "Medium", "Project Experience"),
            InterviewQuestion("Behavioral", "How do you approach learning new technologies or skills?", "Easy", "Learning Ability"),
            InterviewQuestion("Technical", "Describe your experience with the main technologies mentioned in this job description.", "Medium", "Job-Specific Skills"),
            InterviewQuestion("Behavioral", "Tell me about a time you had to solve a problem under pressure.", "Medium", "Problem Solving"),
            InterviewQuestion("Experience", "What's the most complex technical challenge you've faced and how did you overcome it?", "Hard", "Technical Problem Solving"),
            InterviewQuestion("Technical", "How do you ensure code quality and maintainability in your projects?", "Medium", "Best Practices"),
            InterviewQuestion("Behavioral", "Describe a situation where you had to collaborate with team members who had different opinions.", "Medium", "Teamwork"),
            InterviewQuestion("Experience", "Tell me about a time you had to quickly adapt to new requirements or changes.", "Medium", "Adaptability"),
            InterviewQuestion("Technical", "What development tools and methodologies do you prefer and why?", "Easy", "Development Process")
        ]
        questions.extend(fallback_questions[:max(0, num_questions - len(questions))])
    
    return questions[:num_questions] if questions else []

def main():
    # Header
    st.markdown("# 🤖 AI Interview Question Generator")
    st.markdown("**Powered by Google Gemini** - Generate personalized interview questions!")
    st.markdown("---")
    
    # Get API key
    api_key = get_gemini_api_key()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔑 API Status")
        if api_key and api_key != "YOUR_GEMINI_API_KEY_HERE":
            st.success("✅ Gemini API key configured")
        else:
            st.error("❌ Gemini API key not configured")
            st.info("Please configure your API key in secrets.toml")
        
        st.markdown("### ⚙️ Question Settings")
        num_questions = st.slider("Number of Questions", min_value=5, max_value=15, value=10)
        
        st.markdown("### 📊 Question Types")
        st.info("The AI will automatically balance question types based on your inputs")
        
        technical_focus = st.selectbox(
            "Technical Focus Level",
            ["Balanced (50% Technical)", "Technical Heavy (70% Technical)", "Behavioral Heavy (30% Technical)"]
        )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📄 Resume Upload")
        uploaded_resume = st.file_uploader(
            "Upload your resume (PDF)", 
            type=["pdf"],
            help="Upload your resume in PDF format for analysis"
        )
        
        if uploaded_resume:
            st.success("✅ Resume uploaded successfully!")
            
            # Show preview
            with st.expander("📋 Resume Text Preview"):
                resume_text = extract_text_from_pdf(uploaded_resume)
                if resume_text:
                    st.text_area("Extracted text (first 500 characters):", 
                                resume_text[:500] + "..." if len(resume_text) > 500 else resume_text, 
                                height=150, disabled=True)
        
        st.markdown("### 💼 Job Description")
        job_description = st.text_area(
            "Paste the complete job description",
            height=200,
            placeholder="""Paste the full job description here, including:
- Role responsibilities
- Required qualifications
- Preferred skills
- Company information
- Any specific requirements"""
        )
    
    with col2:
        st.markdown("### 🎯 Required Skills & Technologies")
        skills_input = st.text_area(
            "Enter required skills (one per line or comma-separated)",
            height=150,
            placeholder="""Examples:
Python, Django, REST APIs
JavaScript, React, Node.js
SQL, PostgreSQL, MongoDB
AWS, Docker, Kubernetes
Git, CI/CD, Agile
Problem Solving, Communication"""
        )
        
        st.markdown("### 📈 Interview Details")
        
        experience_level = st.selectbox(
            "Your Experience Level",
            ["Entry Level (0-2 years)", "Mid Level (3-5 years)", "Senior Level (6-10 years)", "Lead/Principal (10+ years)"]
        )
        
        interview_type = st.selectbox(
            "Interview Type",
            ["General Technical Interview", "System Design Focus", "Coding Focus", "Behavioral Focus", "Mixed Interview"]
        )
        
        company_info = st.text_input(
            "Company/Industry (Optional)",
            placeholder="e.g., Fintech startup, E-commerce company, Healthcare SaaS"
        )
    
    # Generate button
    st.markdown("---")
    
    # Center the generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button("🚀 Generate Interview Questions", type="primary", use_container_width=True)
    
    if generate_button:
        # Validation
        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            st.error("❌ Gemini API key not configured")
            st.info("💡 Please configure your API key in the secrets.toml file")
            return
        
        if not uploaded_resume:
            st.error("❌ Please upload your resume")
            return
        
        if not job_description.strip():
            st.error("❌ Please enter the job description")
            return
        
        if not skills_input.strip():
            st.error("❌ Please enter the required skills")
            return
        
        # Process inputs
        with st.spinner("📖 Analyzing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_resume)
        
        if not resume_text:
            st.error("❌ Could not extract text from resume. Please ensure it's a readable PDF.")
            return
        
        # Add context to the prompt
        enhanced_prompt = f"""
        Additional Context:
        - Experience Level: {experience_level}
        - Interview Type: {interview_type}
        - Technical Focus: {technical_focus}
        """
        
        if company_info:
            enhanced_prompt += f"\n- Company/Industry: {company_info}"
        
        # Generate questions
        with st.spinner("🤖 Generating personalized questions with Google Gemini..."):
            questions = generate_questions_with_gemini(
                resume_text + enhanced_prompt, job_description, skills_input, num_questions, api_key
            )
        
        if questions:
            st.success(f"✅ Successfully generated {len(questions)} interview questions!")
            
            # Store in session state
            st.session_state.generated_questions = questions
            st.session_state.show_questions = True
        else:
            st.error("❌ Failed to generate questions. Please try again or check your inputs.")
    
    # Display questions if they exist
    if st.session_state.get('show_questions') and st.session_state.get('generated_questions'):
        questions = st.session_state.generated_questions
        
        st.markdown("---")
        st.markdown("## 📝 Your Personalized Interview Questions")
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Questions", len(questions))
        with col2:
            technical_count = len([q for q in questions if q.category == "Technical"])
            st.metric("Technical", technical_count)
        with col3:
            behavioral_count = len([q for q in questions if q.category == "Behavioral"])
            st.metric("Behavioral", behavioral_count)
        with col4:
            experience_count = len([q for q in questions if q.category == "Experience"])
            st.metric("Experience", experience_count)
        
        # Filter controls
        st.markdown("### 🔍 Filter Questions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            difficulty_filter = st.selectbox("🎚️ Difficulty", ["All"] + ["Easy", "Medium", "Hard"])
        with col2:
            category_filter = st.selectbox("📂 Category", ["All"] + list(set([q.category for q in questions])))
        with col3:
            if st.button("🔄 Clear All Filters"):
                st.rerun()
        
        # Apply filters
        filtered_questions = questions
        if difficulty_filter != "All":
            filtered_questions = [q for q in filtered_questions if q.difficulty == difficulty_filter]
        if category_filter != "All":
            filtered_questions = [q for q in filtered_questions if q.category == category_filter]
        
        if len(filtered_questions) != len(questions):
            st.info(f"Showing {len(filtered_questions)} of {len(questions)} questions")
        
        # Display questions
        st.markdown("### 💬 Interview Questions")
        
        for i, question in enumerate(filtered_questions, 1):
            # Icons and colors
            difficulty_colors = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
            category_icons = {"Technical": "⚙️", "Behavioral": "👥", "Experience": "📚"}
            
            difficulty_icon = difficulty_colors.get(question.difficulty, "⚪")
            category_icon = category_icons.get(question.category, "📝")
            
            with st.expander(f"{difficulty_icon} {category_icon} Question {i}: {question.focus_area}"):
                st.markdown(f"**Question:** {question.question}")
                
                # Question details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Category:** {question.category}")
                with col2:
                    st.markdown(f"**Difficulty:** {question.difficulty}")
                with col3:
                    st.markdown(f"**Focus Area:** {question.focus_area}")
                
                # Preparation notes
                notes_key = f"notes_q{i}_{hash(question.question)}"
                st.text_area(
                    "✏️ Your preparation notes and key points:",
                    key=notes_key,
                    height=120,
                    placeholder="""Write your answer preparation here:
• Key points to mention
• Specific examples from your experience
• Technical details to highlight
• STAR method notes (Situation, Task, Action, Result)"""
                )
        
        # Export section
        if filtered_questions:
            st.markdown("---")
            st.markdown("### 📤 Export Your Questions")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Copy formatted text
                if st.button("📋 Copy Questions", use_container_width=True):
                    formatted_text = ""
                    for i, q in enumerate(filtered_questions, 1):
                        formatted_text += f"Q{i}: {q.question}\n"
                        formatted_text += f"Category: {q.category} | Difficulty: {q.difficulty} | Focus: {q.focus_area}\n\n"
                    
                    st.text_area("Copy this text:", formatted_text, height=200)
            
            with col2:
                # JSON export
                questions_json = json.dumps([
                    {
                        "question_number": i+1,
                        "category": q.category,
                        "question": q.question,
                        "difficulty": q.difficulty,
                        "focus_area": q.focus_area
                    }
                    for i, q in enumerate(filtered_questions)
                ], indent=2)
                
                st.download_button(
                    "📁 JSON Format",
                    questions_json,
                    "interview_questions.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                # Text export for printing
                questions_text = f"INTERVIEW PREPARATION QUESTIONS\n"
                questions_text += f"Generated: Today\n"
                questions_text += f"Total Questions: {len(filtered_questions)}\n"
                questions_text += "="*60 + "\n\n"
                
                for i, q in enumerate(filtered_questions, 1):
                    questions_text += f"QUESTION {i}:\n"
                    questions_text += f"{q.question}\n\n"
                    questions_text += f"Details: {q.category} | {q.difficulty} | {q.focus_area}\n\n"
                    questions_text += "PREPARATION NOTES:\n"
                    questions_text += "_" * 40 + "\n\n\n"
                    questions_text += "="*60 + "\n\n"
                
                st.download_button(
                    "📄 Study Guide",
                    questions_text,
                    "interview_study_guide.txt",
                    "text/plain",
                    use_container_width=True
                )
            
            with col4:
                # Clear questions
                if st.button("🗑️ Clear Questions", use_container_width=True):
                    if 'generated_questions' in st.session_state:
                        del st.session_state.generated_questions
                    if 'show_questions' in st.session_state:
                        del st.session_state.show_questions
                    st.rerun()
    
    # Footer with tips
    st.markdown("---")
    with st.expander("💡 Tips for Better Interview Preparation"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📝 Answer Preparation Tips:
            - **Use the STAR method** (Situation, Task, Action, Result)
            - **Prepare specific examples** from your experience
            - **Quantify achievements** with numbers when possible
            - **Practice out loud** with a timer
            - **Research the company** and role thoroughly
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Technical Question Tips:
            - **Explain your thought process** step by step
            - **Ask clarifying questions** before answering
            - **Discuss trade-offs** in your solutions
            - **Be honest** about what you don't know
            - **Show enthusiasm** for learning new things
            """)
    
    # Footer info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    🤖 Powered by Google Gemini | 🔒 Your data is not stored permanently | 💡 Questions generated in real-time
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = []
    if 'show_questions' not in st.session_state:
        st.session_state.show_questions = False
    
    main()