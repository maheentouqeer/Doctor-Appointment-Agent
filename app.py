import json
import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px
import plotly.graph_objects as go
import io
from io import BytesIO

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# CrewAI and LangChain imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Updated LangChain imports (handle version compatibility)
try:
    from langchain_community.llms import HuggingFacePipeline
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain.llms import HuggingFacePipeline
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFacePipeline = None
        HuggingFaceEmbeddings = None

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ========================================================================================
# CONFIGURATION AND SETUP
# ========================================================================================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title="AI Doctor Finder & Booking",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================================
# DATA MODELS
# ========================================================================================

@dataclass
class Doctor:
    id: int
    name: str
    specialty: str
    experience: int
    rating: float
    consultation_fee_pkr: int
    consultation_fee_usd: int
    location: str
    available_slots: List[str]
    contact: str
    hospital: str
    qualification: str = ""
    languages: List[str] = None

@dataclass
class Appointment:
    doctor_id: int
    patient_name: str
    patient_phone: str
    patient_email: str
    appointment_date: str
    appointment_time: str
    total_cost: float
    currency: str
    status: str = "Confirmed"
    booking_id: str = ""

# ========================================================================================
# COMPREHENSIVE DOCTOR DATABASE (60+ DOCTORS) - FIXED VERSION
# ========================================================================================
def generate_comprehensive_doctor_database():
    """Generate a comprehensive database of 150+ doctors with emphasis on Karachi"""
    
    specialties = [
        "Cardiologist", "Dermatologist", "Neurologist", "Pediatrician", 
        "Orthopedic", "Gynecologist", "General Physician", "Psychiatrist", 
        "Urologist", "ENT Specialist", "Oncologist", "Radiologist",
        "Anesthesiologist", "Pathologist", "Gastroenterologist", "Endocrinologist",
        "Pulmonologist", "Nephrologist", "Rheumatologist", "Ophthalmologist",
        "Plastic Surgeon", "Neurosurgeon", "Vascular Surgeon", "Infectious Disease",
        "Emergency Medicine", "Family Medicine", "Internal Medicine", "Hematologist"
    ]
    
    # Expanded locations with emphasis on Karachi
    locations = ["Karachi", "Karachi", "Karachi", "Karachi", "Karachi", "Karachi", 
                "Lahore", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", 
                "Multan", "Peshawar", "Quetta", "Hyderabad", "Gujranwala"]
    
    hospitals = {
        "Karachi": [
            "Aga Khan University Hospital", "Liaquat National Hospital", "Civil Hospital Karachi", 
            "Indus Hospital", "Ziauddin Hospital", "Jinnah Postgraduate Medical Centre",
            "National Institute of Cardiovascular Disease", "Sindh Institute of Urology",
            "Karachi Institute of Heart Disease", "South City Hospital", "Medicare Hospital",
            "Tabba Heart Institute", "Mamji Hospital", "Omar Hospital", "Anklesaria Hospital",
            "Doctors Hospital", "City Hospital", "Hamdard University Hospital", 
            "Baqai Medical University Hospital", "Fazaia Ruth Pfau Medical College Hospital",
            "Shaukat Khanum Memorial Cancer Hospital Karachi", "United Hospital",
            "Hill Park General Hospital", "Lifeline Hospital", "Essa Laboratory Hospital",
            "Al-Tibri Medical College Hospital", "Patel Hospital", "Burhani Hospital",
            "Kharadar General Hospital", "Abbasi Shaheed Hospital", "Qatar Hospital"
        ],
        "Lahore": ["Shaukat Khanum Hospital", "Services Hospital", "Jinnah Hospital", 
                  "Fatima Memorial Hospital", "Hameed Latif Hospital", "Gulab Devi Hospital",
                  "Punjab Institute of Cardiology", "Children Hospital Lahore"],
        "Islamabad": ["PIMS Hospital", "Shifa Hospital", "CDA Hospital", "Federal Government Hospital", 
                     "Maroof Hospital", "Kulsum International Hospital", "Quaid-e-Azam International Hospital"],
        "Rawalpindi": ["Armed Forces Institute", "Benazir Bhutto Hospital", "Holy Family Hospital", 
                      "CMH Rawalpindi", "Fauji Foundation Hospital"],
        "Faisalabad": ["Allied Hospital", "DHQ Hospital", "Children Hospital", "General Hospital",
                      "Aziz Fatimah Hospital"],
        "Multan": ["Nishtar Hospital", "Children Hospital Multan", "General Hospital Multan"],
        "Peshawar": ["Lady Reading Hospital", "Hayatabad Medical Complex", "Khyber Hospital"],
        "Quetta": ["Sandeman Hospital", "Bolan Medical Complex", "Combined Military Hospital"],
        "Hyderabad": ["Liaquat University Hospital", "Government Hospital Hyderabad", "Civil Hospital Hyderabad"],
        "Gujranwala": ["DHQ Hospital Gujranwala", "Aziz Bhatti Shaheed Hospital", "City Hospital Gujranwala"]
    }
    
    # Expanded doctor names (200+ names for variety)
    doctor_names = [
        # Pakistani Male Names
        "Ahmed Khan", "Hassan Shah", "Omar Raza", "Ali Haider", "Bilal Sheikh", "Usman Chaudhry",
        "Tariq Hussain", "Imran Bhatti", "Shahid Iqbal", "Waseem Rehman", "Kashif Hassan",
        "Naveed Baig", "Farhan Gondal", "Salman Khattak", "Adnan Memon", "Kamran Bhatti",
        "Rizwan Iqbal", "Asad Rehman", "Jawad Hassan", "Moiz Baig", "Faisal Gondal",
        "Hamza Khattak", "Danish Memon", "Waqar Bhatti", "Nadeem Rehman", "Rashid Hassan",
        "Ahmed Baig", "Hassan Gondal", "Omar Khattak", "Faisal Ahmed", "Bilal Iqbal",
        "Usman Shah", "Tariq Memon", "Imran Gondal", "Shahid Baig", "Waseem Khattak",
        "Kashif Bhatti", "Naveed Iqbal", "Farhan Shah", "Salman Memon", "Adnan Gondal",
        
        # Pakistani Female Names
        "Fatima Ali", "Ayesha Malik", "Zara Ahmed", "Sana Qureshi", "Mariam Butt",
        "Khadija Siddiqui", "Rabia Awan", "Nadia Dar", "Farah Mirza", "Samina Farooq",
        "Hina Aslam", "Saira Cheema", "Rukhsar Javed", "Bushra Lodhi", "Shazia Niazi",
        "Noor Dar", "Fozia Mirza", "Faiza Farooq", "Rubina Aslam", "Tahira Cheema",
        "Shabana Javed", "Nasreen Lodhi", "Samreen Niazi", "Uzma Dar", "Rehana Mirza",
        "Shehla Farooq", "Yasmeen Aslam", "Fatima Cheema", "Ayesha Javed", "Zara Lodhi",
        "Sana Mirza", "Mariam Farooq", "Khadija Aslam", "Rabia Cheema", "Nadia Javed",
        "Farah Lodhi", "Samina Niazi", "Hina Dar", "Saira Mirza", "Rukhsar Farooq",
        
        # Additional Professional Names
        "Muhammad Asif", "Abdul Rahman", "Syed Ali", "Muhammad Hassan", "Abdul Qadir",
        "Syed Ahmed", "Muhammad Bilal", "Abdul Hamid", "Syed Omar", "Muhammad Tariq",
        "Amna Khan", "Mehreen Shah", "Sadia Malik", "Anum Ahmed", "Sidra Ali",
        "Zainab Hassan", "Hafsa Omar", "Maryam Bilal", "Aisha Tariq", "Saima Asif",
        "Dr. Khalid Mahmood", "Dr. Nasir Ali", "Dr. Sajjad Hussain", "Dr. Rashid Ahmed",
        "Dr. Pervez Iqbal", "Dr. Saeed Shah", "Dr. Naeem Khan", "Dr. Zahid Hassan",
        "Dr. Shafiq Ahmed", "Dr. Arshad Ali", "Dr. Mubashar Shah", "Dr. Tanveer Ahmed",
        "Dr. Amjad Ali", "Dr. Irfan Khan", "Dr. Javed Iqbal", "Dr. Naseem Shah",
        
        # Female Professional Names
        "Dr. Rubina Khatoon", "Dr. Nasreen Akhtar", "Dr. Shahnaz Begum", "Dr. Parveen Akhtar",
        "Dr. Rashida Khatoon", "Dr. Sultana Begum", "Dr. Zahida Parveen", "Dr. Shagufta Nasreen",
        "Dr. Firdous Akhtar", "Dr. Razia Sultana", "Dr. Shamim Akhtar", "Dr. Tahira Nasreen",
        "Dr. Bushra Khatoon", "Dr. Naheed Akhtar", "Dr. Farida Begum", "Dr. Shahida Parveen"
    ]
    
    time_slots = [
        "08:00", "08:30", "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
        "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30",
        "16:00", "16:30", "17:00", "17:30", "18:00", "18:30", "19:00", "19:30"
    ]
    
    # Qualification options
    qualifications = [
        "MBBS, FCPS", "MBBS, FRCS", "MBBS, MD", "MBBS, MS", "MBBS, MCPS", "MBBS, MRCP",
        "MBBS, FCPS, Fellowship", "MBBS, MS, Fellowship", "MBBS, MD, PhD", "MBBS, FRCS, Fellowship"
    ]
    
    doctors = []
    
    # Generate 150 doctors (increased from 60)
    for i in range(1, 151):
        specialty = specialties[i % len(specialties)]
        location = locations[i % len(locations)]
        hospital = hospitals[location][i % len(hospitals[location])]
        name = doctor_names[i % len(doctor_names)]
        
        # Remove "Dr." prefix if it exists to avoid duplication
        if name.startswith("Dr. "):
            name = name[4:]
        
        # Generate realistic fees based on specialty, location, and experience
        base_fee_pkr = random.randint(1500, 5000)
        
        # Specialty-specific fee adjustments
        if specialty in ["Cardiologist", "Neurologist", "Oncologist", "Neurosurgeon"]:
            base_fee_pkr = random.randint(3000, 8000)
        elif specialty in ["Plastic Surgeon", "Vascular Surgeon"]:
            base_fee_pkr = random.randint(4000, 10000)
        elif specialty in ["General Physician", "Family Medicine"]:
            base_fee_pkr = random.randint(1000, 3000)
        elif specialty in ["Dermatologist", "ENT Specialist", "Ophthalmologist"]:
            base_fee_pkr = random.randint(2000, 5000)
        elif specialty in ["Pediatrician", "Gynecologist"]:
            base_fee_pkr = random.randint(1500, 4000)
        
        # Location-specific adjustments (Karachi and Lahore typically higher)
        if location in ["Karachi", "Lahore"]:
            base_fee_pkr = int(base_fee_pkr * random.uniform(1.1, 1.3))
        elif location in ["Islamabad", "Rawalpindi"]:
            base_fee_pkr = int(base_fee_pkr * random.uniform(1.0, 1.2))
        else:
            base_fee_pkr = int(base_fee_pkr * random.uniform(0.8, 1.0))
        
        experience = random.randint(3, 30)
        # Experience-based fee adjustment
        if experience > 20:
            base_fee_pkr = int(base_fee_pkr * random.uniform(1.2, 1.5))
        elif experience > 15:
            base_fee_pkr = int(base_fee_pkr * random.uniform(1.1, 1.3))
        elif experience > 10:
            base_fee_pkr = int(base_fee_pkr * random.uniform(1.0, 1.2))
        
        # Ensure minimum and maximum bounds
        base_fee_pkr = max(1000, min(base_fee_pkr, 12000))
        
        # Rating based on experience and specialization
        if experience > 15 and specialty in ["Cardiologist", "Neurologist", "Oncologist"]:
            rating = round(random.uniform(4.2, 5.0), 1)
        elif experience > 10:
            rating = round(random.uniform(3.8, 4.8), 1)
        else:
            rating = round(random.uniform(3.5, 4.5), 1)
        
        # Generate available slots (4-10 slots per doctor)
        num_slots = random.randint(4, 10)
        available_slots = random.sample(time_slots, num_slots)
        available_slots.sort()
        
        # Generate qualification
        qualification = f"MBBS, {random.choice(['FCPS', 'FRCS', 'MD', 'MS', 'MCPS'])} ({specialty})"
        if experience > 15:
            qualification += f", {random.choice(['Fellowship', 'PhD', 'FRCP'])}"
        
        # Language selection based on location
        if location == "Karachi":
            languages = random.sample(["Urdu", "English", "Sindhi"], random.randint(2, 3))
        elif location == "Lahore":
            languages = random.sample(["Urdu", "English", "Punjabi"], random.randint(2, 3))
        elif location == "Peshawar":
            languages = random.sample(["Urdu", "English", "Pashto"], random.randint(2, 3))
        else:
            languages = random.sample(["Urdu", "English", "Punjabi", "Sindhi"], random.randint(2, 3))
        
        doctor = Doctor(
            id=i,
            name=name,
            specialty=specialty,
            experience=experience,
            rating=rating,
            consultation_fee_pkr=base_fee_pkr,
            consultation_fee_usd=int(base_fee_pkr / 280),
            location=location,
            available_slots=available_slots.copy(),
            contact=f"+92-{random.randint(300, 345)}-{random.randint(1000000, 9999999)}",
            hospital=hospital,
            qualification=qualification,
            languages=languages
        )
        
        doctors.append(doctor)
    
    return doctors
# Initialize comprehensive doctor database
DOCTORS_DATABASE = generate_comprehensive_doctor_database()

# ========================================================================================
# LLM SETUP (Open Source) - Updated for compatibility
# ========================================================================================

class SimpleLLMWrapper:
    """Simple LLM wrapper for CrewAI compatibility"""
    def __init__(self, pipeline=None):
        self.pipeline = pipeline
    
    def __call__(self, prompt):
        try:
            if self.pipeline:
                if isinstance(prompt, list):
                    prompt = " ".join(str(p) for p in prompt)
                
                response = self.pipeline(str(prompt), max_length=200, num_return_sequences=1, pad_token_id=50256)
                
                if response and len(response) > 0:
                    generated_text = response[0]['generated_text']
                    if generated_text.startswith(str(prompt)):
                        return generated_text[len(str(prompt)):].strip()
                    return generated_text.strip()
            
            # Fallback responses based on prompt content
            prompt_str = str(prompt).lower()
            if "search" in prompt_str and "doctor" in prompt_str:
                return "I'll search for doctors matching your criteria including budget, specialty, and location preferences."
            elif "book" in prompt_str and "appointment" in prompt_str:
                return "I'll proceed with booking the appointment using the provided patient details and selected time slot."
            elif "form" in prompt_str and "submit" in prompt_str:
                return "I'll submit the appointment form and generate a confirmation number for your booking."
            else:
                return f"I understand your request. I'll help you accordingly."
                
        except Exception as e:
            return f"Processing your request..."
    
    def invoke(self, prompt):
        return self.__call__(prompt)

@st.cache_resource
def load_open_source_llm():
    """Load open-source LLM model with enhanced fallback"""
    try:
        st.info("Loading AI model...")
        return SimpleLLMWrapper()
    except Exception as e:
        st.warning(f"Using mock LLM for demonstration: {str(e)}")
        return SimpleLLMWrapper()

# ========================================================================================
# PDF GENERATION FUNCTIONS - NEW
# ========================================================================================

def generate_pdf_receipt(booking_details):
    """Generate professional PDF receipt"""
    if not PDF_AVAILABLE:
        return None
    
    buffer = BytesIO()
    
    try:
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
        
        # Get sample stylesheet
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkgreen
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=6
        )
        
        # Story array to hold content
        story = []
        
        # Title
        story.append(Paragraph("🏥 MEDICAL APPOINTMENT RECEIPT", title_style))
        story.append(Spacer(1, 20))
        
        # Booking ID section
        story.append(Paragraph("📋 BOOKING INFORMATION", header_style))
        
        booking_data = [
            ['Booking ID:', booking_details['booking_id']],
            ['Booking Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Status:', booking_details['status']]
        ]
        
        booking_table = Table(booking_data, colWidths=[2*inch, 4*inch])
        booking_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(booking_table)
        story.append(Spacer(1, 20))
        
        # Patient Information
        story.append(Paragraph("👤 PATIENT INFORMATION", header_style))
        
        patient_data = [
            ['Patient Name:', booking_details['patient_name']],
            ['Phone Number:', booking_details.get('patient_phone', 'N/A')],
            ['Email Address:', booking_details.get('patient_email', 'N/A')]
        ]
        
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        # Doctor Information
        story.append(Paragraph("👨‍⚕️ DOCTOR INFORMATION", header_style))
        
        doctor_data = [
            ['Doctor Name:', f"Dr. {booking_details['doctor_name']}"],
            ['Specialty:', booking_details['specialty']],
            ['Hospital:', booking_details['hospital']],
            ['Contact:', booking_details.get('doctor_contact', 'N/A')]
        ]
        
        doctor_table = Table(doctor_data, colWidths=[2*inch, 4*inch])
        doctor_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(doctor_table)
        story.append(Spacer(1, 20))
        
        # Appointment Details
        story.append(Paragraph("📅 APPOINTMENT DETAILS", header_style))
        
        appointment_data = [
            ['Date:', booking_details['appointment_date']],
            ['Time:', booking_details['appointment_time']],
            ['Fee:', f"{booking_details['total_cost']} {booking_details['currency']}"]
        ]
        
        appointment_table = Table(appointment_data, colWidths=[2*inch, 4*inch])
        appointment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(appointment_table)
        story.append(Spacer(1, 30))
        
        # Important Instructions
        story.append(Paragraph("📋 IMPORTANT INSTRUCTIONS", header_style))
        
        instructions = [
            "• Please arrive 15 minutes before your appointment time",
            "• Bring a valid photo ID for verification",
            "• Carry any previous medical records or test reports",
            "• Contact the hospital if you need to reschedule",
            "• Keep this receipt for your records"
        ]
        
        for instruction in instructions:
            story.append(Paragraph(instruction, normal_style))
        
        story.append(Spacer(1, 30))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=1,
            textColor=colors.grey
        )
        
        story.append(Paragraph("Thank you for choosing our medical booking service!", footer_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
        
    except Exception as e:
        print(f"PDF generation error: {str(e)}")
        return None

# ========================================================================================
# CREWAI TOOLS - FIXED SEARCH LOGIC
# ========================================================================================

class AdvancedDoctorSearchTool(BaseTool):
    name: str = "Advanced Doctor Search Tool"
    description: str = "Advanced search with comprehensive filters"
    
    def _run(self, budget: float, currency: str, specialty: str = "", location: str = "",
             min_rating: float = 0.0, min_experience: int = 0, preferred_time: str = "",
             sort_by: str = "rating") -> str:
        """FIXED: Enhanced search with all filters working correctly"""
        try:
            matching_doctors = []
            
            # Debug print for search parameters
            print(f"DEBUG: Search Parameters:")
            print(f"  Specialty: '{specialty}'")
            print(f"  Location: '{location}'")
            print(f"  Budget: {budget} {currency}")
            print(f"  Min Rating: {min_rating}")
            print(f"  Min Experience: {min_experience}")
            
            for doctor in DOCTORS_DATABASE:
                doctor_fee = doctor.consultation_fee_pkr if currency.upper() == "PKR" else doctor.consultation_fee_usd
                
                # Apply budget filter
                if doctor_fee > budget:
                    continue
                
                # Apply rating filter
                if doctor.rating < min_rating:
                    continue
                    
                # Apply experience filter
                if doctor.experience < min_experience:
                    continue
                
                # FIXED: Apply specialty filter - exact match, case insensitive
                if specialty and specialty.strip():
                    if specialty.lower().strip() != doctor.specialty.lower().strip():
                        continue
                
                # Apply location filter - exact match, case insensitive
                if location and location.strip():
                    if location.lower().strip() != doctor.location.lower().strip():
                        continue
                
                # Apply time filter
                if preferred_time and preferred_time.strip():
                    if preferred_time not in doctor.available_slots:
                        continue
                
                # Add matching doctor with FIXED data consistency
                matching_doctors.append({
                    "id": doctor.id,  # FIXED: Use actual doctor ID
                    "name": doctor.name,
                    "specialty": doctor.specialty,
                    "experience": doctor.experience,
                    "rating": doctor.rating,
                    "fee": doctor_fee,
                    "location": doctor.location,
                    "hospital": doctor.hospital,
                    "contact": doctor.contact,
                    "qualification": doctor.qualification,
                    "languages": doctor.languages or ["Urdu", "English"],
                    "available_slots": doctor.available_slots.copy()  # FIXED: Use copy
                })
            
            # Apply sorting
            if sort_by == "rating":
                matching_doctors.sort(key=lambda x: x['rating'], reverse=True)
            elif sort_by == "experience":
                matching_doctors.sort(key=lambda x: x['experience'], reverse=True)
            elif sort_by == "price_low":
                matching_doctors.sort(key=lambda x: x['fee'])
            elif sort_by == "price_high":
                matching_doctors.sort(key=lambda x: x['fee'], reverse=True)
            
            print(f"DEBUG: Found {len(matching_doctors)} doctors matching criteria")
            
            if not matching_doctors:
                return json.dumps({"error": "No doctors found matching all criteria"})
            
            return json.dumps({
                "success": True,
                "count": len(matching_doctors),
                "doctors": matching_doctors,
                "search_params": {
                    "specialty": specialty,
                    "location": location,
                    "budget": budget,
                    "currency": currency,
                    "min_rating": min_rating,
                    "min_experience": min_experience,
                    "preferred_time": preferred_time,
                    "sort_by": sort_by
                }
            }, indent=2)
            
        except Exception as e:
            print(f"DEBUG: Search error: {str(e)}")
            return json.dumps({"error": f"Search error: {str(e)}"})

class AppointmentBookingTool(BaseTool):
    name: str = "Appointment Booking Tool"
    description: str = "Book appointments with comprehensive validation"
    
    def _run(self, doctor_id: int, patient_name: str, patient_phone: str, 
             patient_email: str, appointment_date: str, appointment_time: str,
             currency: str) -> str:
        """FIXED: Enhanced booking with proper doctor ID handling"""
        try:
            # FIXED: Find doctor by exact ID match
            doctor = None
            for d in DOCTORS_DATABASE:
                if d.id == doctor_id:
                    doctor = d
                    break
            
            if not doctor:
                return json.dumps({"error": f"Doctor with ID {doctor_id} not found"})
            
            # Validate slot availability
            if appointment_time not in doctor.available_slots:
                return json.dumps({
                    "error": f"Time slot {appointment_time} not available",
                    "available_slots": doctor.available_slots
                })
            
            # Calculate cost
            total_cost = doctor.consultation_fee_pkr if currency.upper() == "PKR" else doctor.consultation_fee_usd
            
            # Generate booking ID
            booking_id = f"BK{random.randint(100000, 999999)}"
            
            # Create appointment with FIXED doctor data
            appointment = Appointment(
                doctor_id=doctor.id,  # FIXED: Use correct doctor ID
                patient_name=patient_name,
                patient_phone=patient_phone,
                patient_email=patient_email,
                appointment_date=appointment_date,
                appointment_time=appointment_time,
                total_cost=total_cost,
                currency=currency,
                booking_id=booking_id
            )
            
            # Store in session state
            if 'appointments' not in st.session_state:
                st.session_state.appointments = []
            st.session_state.appointments.append(appointment)
            
            # FIXED: Remove booked slot from the correct doctor
            if appointment_time in doctor.available_slots:
                doctor.available_slots.remove(appointment_time)
            
            # FIXED: Return booking details with consistent doctor information
            booking_details = {
                "success": True,
                "booking_id": booking_id,
                "doctor_name": doctor.name,  # FIXED: Use correct doctor name
                "specialty": doctor.specialty,  # FIXED: Use correct specialty
                "hospital": doctor.hospital,  # FIXED: Use correct hospital
                "patient_name": patient_name,
                "patient_phone": patient_phone,
                "patient_email": patient_email,
                "appointment_date": appointment_date,
                "appointment_time": appointment_time,
                "total_cost": total_cost,
                "currency": currency,
                "status": "Confirmed",
                "doctor_contact": doctor.contact  # FIXED: Use correct contact
            }
            
            return json.dumps(booking_details, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Booking failed: {str(e)}"})

# ========================================================================================
# STREAMLIT UI COMPONENTS - FIXED
# ========================================================================================

def display_doctor_card(doctor_data, currency="PKR"):
    """FIXED: Enhanced doctor card display with working booking button"""
    with st.container():
        # Create card-like appearance with border
        st.markdown("""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"### Dr. {doctor_data['name']}")
            st.markdown(f"**{doctor_data['specialty']}**")
            st.markdown(f"📍 {doctor_data['location']} • {doctor_data['hospital']}")
            st.markdown(f"🎓 {doctor_data.get('qualification', 'MBBS')}")
            
            # Languages
            if 'languages' in doctor_data and doctor_data['languages']:
                languages_str = ", ".join(doctor_data['languages'])
                st.markdown(f"🗣️ {languages_str}")
        
        with col2:
            # Rating with stars
            stars = "⭐" * int(doctor_data['rating'])
            st.markdown(f"**Rating:** {stars} ({doctor_data['rating']}/5)")
            st.markdown(f"**Experience:** {doctor_data['experience']} years")
            st.markdown(f"**Fee:** {doctor_data['fee']} {currency}")
            st.markdown(f"📞 {doctor_data.get('contact', 'N/A')}")
        
        with col3:
            # FIXED: Unique button key and proper callback
            button_key = f"book_{doctor_data['id']}_{hash(str(doctor_data['name']))}"
            
            if st.button("📅 Book Now", key=button_key, type="primary", use_container_width=True):
                # FIXED: Store selected doctor data with proper ID
                st.session_state.selected_doctor_for_booking = doctor_data.copy()
                st.session_state.booking_step = 'patient_form'
                
                # Show immediate feedback
                st.success(f"✅ Selected Dr. {doctor_data['name']} for booking!")
                st.info("📋 Please fill in your details below to complete the booking.")
                
                # Auto-scroll to booking form
                st.rerun()
        
        # Available slots with better formatting
        st.markdown("**Available Time Slots:**")
        if 'available_slots' in doctor_data and doctor_data['available_slots']:
            # Display slots in a grid format
            slots = doctor_data['available_slots']
            slots_per_row = 6
            
            # Create rows of slots
            for i in range(0, len(slots), slots_per_row):
                slot_row = slots[i:i+slots_per_row]
                cols = st.columns(len(slot_row))
                
                for j, slot in enumerate(slot_row):
                    with cols[j]:
                        st.markdown(f"`{slot}`")
        else:
            st.markdown("No slots available")
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.divider()

def show_patient_booking_form(doctor_data):
    """FIXED: Show inline booking form when a doctor is selected - MOVED OUTSIDE FORM"""
    st.markdown("---")
    st.header("📋 Book Appointment")
    
    # Display selected doctor summary
    with st.expander("Selected Doctor Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Doctor:** Dr. {doctor_data['name']}")
            st.write(f"**Specialty:** {doctor_data['specialty']}")
            st.write(f"**Hospital:** {doctor_data['hospital']}")
            st.write(f"**Location:** {doctor_data['location']}")
        with col2:
            st.write(f"**Fee:** {doctor_data['fee']} {st.session_state.get('search_currency', 'PKR')}")
            st.write(f"**Rating:** {doctor_data['rating']}/5 ⭐")
            st.write(f"**Experience:** {doctor_data['experience']} years")
            st.write(f"**Contact:** {doctor_data['contact']}")
    
    # Patient details form
    with st.form("patient_booking_form", clear_on_submit=False):
        st.subheader("👤 Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Full Name *", placeholder="Enter your full name")
            patient_phone = st.text_input("Phone Number *", placeholder="+92-300-1234567")
            patient_email = st.text_input("Email Address *", placeholder="patient@email.com")
        
        with col2:
            # Date selection
            min_date = datetime.now().date()
            max_date = min_date + timedelta(days=30)
            appointment_date = st.date_input("Preferred Date *", 
                                           min_value=min_date,
                                           max_value=max_date,
                                           value=min_date)
            
            # Time selection
            appointment_time = st.selectbox("Preferred Time *", doctor_data['available_slots'])
            
            # Additional reason
            reason = st.text_area("Reason for Visit (Optional)", 
                                placeholder="Brief description of your concern...")
        
        # Terms and conditions
        agree_terms = st.checkbox("I agree to the terms and conditions *")
        
        # Submit button
        col1, col2 = st.columns([1, 2])
        with col1:
            submitted = st.form_submit_button("📅 Book Appointment", type="primary", use_container_width=True)
        with col2:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                if 'selected_doctor_for_booking' in st.session_state:
                    del st.session_state.selected_doctor_for_booking
                if 'booking_step' in st.session_state:
                    del st.session_state.booking_step
                st.success("Booking cancelled.")
                st.rerun()
        
        if submitted:
            # Validation
            if not all([patient_name.strip(), patient_phone.strip(), patient_email.strip(), agree_terms]):
                st.error("❌ Please fill all required fields and agree to terms")
                return False
            
            # Email validation
            if "@" not in patient_email or "." not in patient_email:
                st.error("❌ Please enter a valid email address")
                return False
            
            # Phone validation
            if len(patient_phone.replace("-", "").replace("+", "")) < 10:
                st.error("❌ Please enter a valid phone number")
                return False
            
            # Process booking
            with st.spinner("🤖 Booking your appointment..."):
                booking_tool = AppointmentBookingTool()
                
                result = booking_tool._run(
                    doctor_id=doctor_data['id'],  # FIXED: Use correct doctor ID
                    patient_name=patient_name,
                    patient_phone=patient_phone,
                    patient_email=patient_email,
                    appointment_date=appointment_date.strftime("%Y-%m-%d"),
                    appointment_time=appointment_time,
                    currency=st.session_state.get('search_currency', 'PKR')
                )
                
                try:
                    booking_result = json.loads(result)
                    
                    if booking_result.get("success"):
                        # Clear booking form state
                        if 'selected_doctor_for_booking' in st.session_state:
                            del st.session_state.selected_doctor_for_booking
                        if 'booking_step' in st.session_state:
                            del st.session_state.booking_step
                        
                        # Show success message
                        st.success("🎉 Appointment booked successfully!")
                        
                        # Store booking result for receipt generation
                        st.session_state.last_booking = booking_result
                        
                        return True
                        
                    else:
                        st.error(f"❌ {booking_result.get('error', 'Booking failed')}")
                        return False
                        
                except Exception as e:
                    st.error(f"❌ Booking error: {str(e)}")
                    return False
    
    # FIXED: Show booking confirmation OUTSIDE the form
    if 'last_booking' in st.session_state:
        show_booking_confirmation(st.session_state.last_booking)
        # Clear the last booking after showing
        del st.session_state.last_booking

def show_booking_confirmation(booking_result):
    """FIXED: Show booking confirmation with proper receipt download - OUTSIDE FORM"""
    st.markdown("### ✅ Booking Confirmation")
    
    with st.container():
        st.markdown("""
        <div style="border: 2px solid #28a745; border-radius: 15px; padding: 25px; margin: 15px 0; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **🆔 Booking ID:** `{booking_result['booking_id']}`  
            **👤 Patient:** {booking_result['patient_name']}  
            **👨‍⚕️ Doctor:** Dr. {booking_result['doctor_name']}  
            **🏥 Specialty:** {booking_result['specialty']}  
            **🏢 Hospital:** {booking_result['hospital']}
            """)
        
        with col2:
            st.markdown(f"""
            **📅 Date:** {booking_result['appointment_date']}  
            **🕐 Time:** {booking_result['appointment_time']}  
            **💰 Fee:** {booking_result['total_cost']} {booking_result['currency']}  
            **📊 Status:** {booking_result['status']}  
            **📞 Contact:** {booking_result.get('doctor_contact', 'N/A')}
            """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # FIXED: Download buttons OUTSIDE the form
    col1, col2 = st.columns(2)
    
    with col1:
        # Text receipt
        receipt_data = f"""
APPOINTMENT RECEIPT
==================
Booking ID: {booking_result['booking_id']}
Patient: {booking_result['patient_name']}
Doctor: Dr. {booking_result['doctor_name']}
Specialty: {booking_result['specialty']}
Hospital: {booking_result['hospital']}
Date: {booking_result['appointment_date']}
Time: {booking_result['appointment_time']}
Fee: {booking_result['total_cost']} {booking_result['currency']}
Status: {booking_result['status']}
Doctor Contact: {booking_result.get('doctor_contact', 'N/A')}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        st.download_button(
            "📋 Download Text Receipt",
            receipt_data,
            f"receipt_{booking_result['booking_id']}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # FIXED: PDF receipt generation
        if PDF_AVAILABLE:
            pdf_buffer = generate_pdf_receipt(booking_result)
            if pdf_buffer:
                st.download_button(
                    "📄 Download PDF Receipt",
                    pdf_buffer.getvalue(),
                    f"receipt_{booking_result['booking_id']}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.button("📄 PDF Unavailable", disabled=True, use_container_width=True)
        else:
            st.button("📄 Install reportlab for PDF", disabled=True, use_container_width=True)
    
    # Show important instructions
    st.info(f"""
    📋 **Important Instructions:**
    - Arrive 15 minutes before your appointment time
    - Bring a valid ID and any previous medical records  
    - Contact the hospital at {booking_result.get('doctor_contact', 'N/A')} if you need to reschedule
    - Your booking ID is: **{booking_result['booking_id']}** (save this for reference)
    """)

# ========================================================================================
# MAIN APPLICATION INTERFACES - FIXED
# ========================================================================================

def advanced_search_interface():
    """FIXED: Enhanced advanced search interface with working specialty filter"""
    st.header("🔬 Advanced Doctor Search")
    st.markdown("Find the perfect doctor with comprehensive filters")
    
    # Initialize search form state if not exists
    if 'search_form_state' not in st.session_state:
        st.session_state.search_form_state = {
            'budget': 5000,
            'currency': 'PKR',
            'specialty': 'All Specialties',
            'location': 'All Locations',
            'min_rating': 4.0,
            'min_experience': 5,
            'preferred_time': 'Any Time',
            'sort_by': 'rating'
        }
    
    # FIXED: Use form with proper state management
    with st.form("advanced_search_form", clear_on_submit=False):
        # Budget and currency row
        col1, col2 = st.columns(2)
        with col1:
            budget = st.slider("💰 Maximum Budget", 1000, 10000, 
                             value=st.session_state.search_form_state['budget'], 
                             step=500)
        with col2:
            currency = st.selectbox("💱 Currency", ["PKR", "USD"], 
                                  index=0 if st.session_state.search_form_state['currency'] == 'PKR' else 1)
        
        # Filters row 1
        col3, col4, col5 = st.columns(3)
        with col3:
            # FIXED: Get unique specialties and handle selection properly
            all_specialties = sorted(list(set([doc.specialty for doc in DOCTORS_DATABASE])))
            specialty_options = ["All Specialties"] + all_specialties
            
            # Find current specialty index
            current_specialty = st.session_state.search_form_state['specialty']
            try:
                specialty_index = specialty_options.index(current_specialty)
            except ValueError:
                specialty_index = 0
            
            specialty = st.selectbox("🏥 Specialty", specialty_options, index=specialty_index)
            
        with col4:
            # Get unique locations
            all_locations = sorted(list(set([doc.location for doc in DOCTORS_DATABASE])))
            location_options = ["All Locations"] + all_locations
            
            # Find current location index
            current_location = st.session_state.search_form_state['location']
            try:
                location_index = location_options.index(current_location)
            except ValueError:
                location_index = 0
                    
            location = st.selectbox("📍 Location", location_options, index=location_index)
            
        with col5:
            min_rating = st.slider("⭐ Minimum Rating", 0.0, 5.0, 
                                 value=st.session_state.search_form_state['min_rating'], 
                                 step=0.1)
        
        # Filters row 2
        col6, col7, col8 = st.columns(3)
        with col6:
            min_experience = st.slider("👨‍⚕️ Min Experience (years)", 0, 25, 
                                     value=st.session_state.search_form_state['min_experience'])
        with col7:
            all_time_slots = sorted(list(set([slot for doc in DOCTORS_DATABASE for slot in doc.available_slots])))
            time_options = ["Any Time"] + all_time_slots
            
            # Find current time index
            current_time = st.session_state.search_form_state['preferred_time']
            try:
                time_index = time_options.index(current_time)
            except ValueError:
                time_index = 0
                    
            preferred_time = st.selectbox("🕐 Preferred Time", time_options, index=time_index)
            
        with col8:
            sort_options = [
                ("rating", "Rating (High to Low)"),
                ("experience", "Experience (High to Low)"),
                ("price_low", "Price (Low to High)"),
                ("price_high", "Price (High to Low)")
            ]
            
            # Find current sort index
            current_sort = st.session_state.search_form_state['sort_by']
            sort_index = next((i for i, (key, _) in enumerate(sort_options) if key == current_sort), 0)
            
            sort_selection = st.selectbox("📊 Sort By", sort_options,
                                        index=sort_index,
                                        format_func=lambda x: x[1])
            sort_by = sort_selection[0]
        
        # Search button
        search_submitted = st.form_submit_button("🔍 Search Doctors", use_container_width=True, type="primary")
        
        if search_submitted:
            # Update form state
            st.session_state.search_form_state = {
                'budget': budget,
                'currency': currency,
                'specialty': specialty,
                'location': location,
                'min_rating': min_rating,
                'min_experience': min_experience,
                'preferred_time': preferred_time,
                'sort_by': sort_by
            }
            
            # FIXED: Convert "All" options to empty strings for the search
            search_specialty = "" if specialty == "All Specialties" else specialty
            search_location = "" if location == "All Locations" else location
            search_time = "" if preferred_time == "Any Time" else preferred_time
            
            # Store search parameters for display
            st.session_state.last_search_params = {
                'budget': budget,
                'currency': currency,
                'specialty': specialty,  # Store original for display
                'location': location,    # Store original for display
                'min_rating': min_rating,
                'min_experience': min_experience,
                'preferred_time': preferred_time,  # Store original for display
                'sort_by': sort_by
            }
            
            # Store currency for booking
            st.session_state.search_currency = currency
            
            with st.spinner("🤖 AI agents are searching for doctors..."):
                search_tool = AdvancedDoctorSearchTool()
                results = search_tool._run(
                    budget=budget,
                    currency=currency,
                    specialty=search_specialty,
                    location=search_location,
                    min_rating=min_rating,
                    min_experience=min_experience,
                    preferred_time=search_time,
                    sort_by=sort_by
                )
                
                try:
                    search_result = json.loads(results)
                    
                    if search_result.get("success"):
                        doctors_data = search_result["doctors"]
                        st.session_state.search_results = doctors_data
                        st.success(f"✅ Found {len(doctors_data)} doctors matching your criteria!")
                        
                        # Clear any previous booking state
                        if 'selected_doctor_for_booking' in st.session_state:
                            del st.session_state.selected_doctor_for_booking
                        if 'booking_step' in st.session_state:
                            del st.session_state.booking_step
                        
                        # Force rerun to show results
                        st.rerun()
                    else:
                        st.error(f"❌ {search_result.get('error', 'No doctors found matching your criteria')}")
                        st.info("💡 Try adjusting your search criteria (budget, rating, experience, etc.)")
                        
                except Exception as e:
                    st.error(f"❌ Error processing search: {str(e)}")

def display_search_results():
    """Display search results with proper parameters"""
    if 'search_results' not in st.session_state or not st.session_state.search_results:
        return False
    
    results = st.session_state.search_results
    search_params = st.session_state.get('last_search_params', {})
    currency = st.session_state.get('search_currency', 'PKR')
    
    st.header("🏥 Search Results")
    st.success(f"✅ Found {len(results)} doctor(s) matching your criteria")
    
    # FIXED: Display search criteria with proper values
    with st.expander("🔍 Search Criteria Used", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**Budget:** {search_params.get('budget', 'N/A')} {currency}")
            st.write(f"**Min Rating:** {search_params.get('min_rating', 'N/A')}⭐")
        with col2:
            # FIXED: Show the actual specialty selected
            specialty_display = search_params.get('specialty', 'Any')
            st.write(f"**Specialty:** {specialty_display}")
            st.write(f"**Min Experience:** {search_params.get('min_experience', 'N/A')} years")
        with col3:
            location_display = search_params.get('location', 'Any')
            st.write(f"**Location:** {location_display}")
            time_display = search_params.get('preferred_time', 'Any')
            st.write(f"**Preferred Time:** {time_display}")
        with col4:
            sort_display = {
                'rating': 'Rating (High to Low)',
                'experience': 'Experience (High to Low)',
                'price_low': 'Price (Low to High)',
                'price_high': 'Price (High to Low)'
            }.get(search_params.get('sort_by', 'rating'), 'Rating')
            st.write(f"**Sorted by:** {sort_display}")
    
    st.markdown("---")
    st.markdown("### 👨‍⚕️ Available Doctors")
    
    # Display each doctor
    for i, doctor in enumerate(results, 1):
        st.markdown(f"#### {i}. Dr. {doctor['name']}")
        display_doctor_card(doctor, currency)
    
    return True

def appointments_interface():
    """FIXED: Enhanced appointments view with proper doctor data consistency"""
    st.header("📅 My Appointments")
    
    if 'appointments' not in st.session_state or not st.session_state.appointments:
        st.info("📭 No appointments found")
        st.markdown("Book your first appointment to see it here!")
        if st.button("📅 Book New Appointment", use_container_width=True, type="primary"):
            # Clear any booking state and go to search
            for key in ['selected_doctor_for_booking', 'booking_step', 'search_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return
    
    appointments = st.session_state.appointments
    st.success(f"📊 You have {len(appointments)} appointment(s)")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Filter by Status", ["All", "Confirmed", "Cancelled", "Completed"])
    with col2:
        date_filter = st.selectbox("Filter by Date", ["All", "Upcoming", "Past"])
    with col3:
        sort_option = st.selectbox("Sort by", ["Date (Latest)", "Date (Oldest)", "Doctor Name"])
    
    # Apply filters
    filtered_appointments = appointments.copy()
    
    if status_filter != "All":
        filtered_appointments = [apt for apt in filtered_appointments if apt.status == status_filter]
    
    today = datetime.now().date()
    if date_filter == "Upcoming":
        filtered_appointments = [apt for apt in filtered_appointments 
                               if datetime.strptime(apt.appointment_date, "%Y-%m-%d").date() >= today]
    elif date_filter == "Past":
        filtered_appointments = [apt for apt in filtered_appointments 
                               if datetime.strptime(apt.appointment_date, "%Y-%m-%d").date() < today]
    
    # Sort appointments
    if sort_option == "Date (Latest)":
        filtered_appointments.sort(key=lambda x: x.appointment_date, reverse=True)
    elif sort_option == "Date (Oldest)":
        filtered_appointments.sort(key=lambda x: x.appointment_date)
    elif sort_option == "Doctor Name":
        filtered_appointments.sort(key=lambda x: next((d.name for d in DOCTORS_DATABASE if d.id == x.doctor_id), ""))
    
    # Display appointments
    if not filtered_appointments:
        st.warning("⚠️ No appointments match your current filters.")
        return
    
    for i, appointment in enumerate(filtered_appointments, 1):
        # FIXED: Find doctor by exact ID match for consistency
        doctor = None
        for d in DOCTORS_DATABASE:
            if d.id == appointment.doctor_id:
                doctor = d
                break
        
        # Status emoji
        status_emoji = "✅" if appointment.status == "Confirmed" else "❌" if appointment.status == "Cancelled" else "✔️"
        
        with st.expander(f"{status_emoji} Appointment #{i} - Dr. {doctor.name if doctor else 'Unknown'} ({appointment.status})", 
                        expanded=i==1):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                **👤 Patient:** {appointment.patient_name}  
                **👨‍⚕️ Doctor:** Dr. {doctor.name if doctor else 'Unknown'}  
                **🏥 Specialty:** {doctor.specialty if doctor else 'N/A'}  
                **🆔 Booking ID:** `{appointment.booking_id}`
                """)
            
            with col2:
                st.markdown(f"""
                **📅 Date:** {appointment.appointment_date}  
                **🕐 Time:** {appointment.appointment_time}  
                **💰 Fee:** {appointment.total_cost} {appointment.currency}  
                **📊 Status:** {appointment.status}
                """)
            
            with col3:
                st.markdown(f"""
                **🏢 Hospital:** {doctor.hospital if doctor else 'N/A'}  
                **📞 Doctor Contact:** {doctor.contact if doctor else 'N/A'}  
                **📱 Your Phone:** {appointment.patient_phone}  
                **📧 Your Email:** {appointment.patient_email}
                """)
            
            # Action buttons for each appointment
            button_col1, button_col2, button_col3, button_col4 = st.columns(4)
            
            with button_col1:
                if appointment.status == "Confirmed":
                    if st.button(f"❌ Cancel", key=f"cancel_{appointment.booking_id}"):
                        appointment.status = "Cancelled"
                        st.success("✅ Appointment cancelled successfully!")
                        st.rerun()
            
            with button_col2:
                if st.button(f"📧 Send Reminder", key=f"remind_{appointment.booking_id}"):
                    st.success("✅ Reminder sent to your email and phone!")
            
            with button_col3:
                # Download text receipt
                receipt_data = f"""
APPOINTMENT RECEIPT
==================
Booking ID: {appointment.booking_id}
Patient: {appointment.patient_name}
Doctor: Dr. {doctor.name if doctor else 'Unknown'}
Specialty: {doctor.specialty if doctor else 'N/A'}
Hospital: {doctor.hospital if doctor else 'N/A'}
Date: {appointment.appointment_date}
Time: {appointment.appointment_time}
Fee: {appointment.total_cost} {appointment.currency}
Status: {appointment.status}
Doctor Contact: {doctor.contact if doctor else 'N/A'}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                st.download_button(
                    "📋 Text Receipt",
                    receipt_data,
                    f"receipt_{appointment.booking_id}.txt",
                    mime="text/plain",
                    key=f"dl_txt_{appointment.booking_id}"
                )
            
            with button_col4:
                # FIXED: PDF receipt with consistent doctor data
                if PDF_AVAILABLE and doctor:
                    booking_details = {
                        "booking_id": appointment.booking_id,
                        "patient_name": appointment.patient_name,
                        "patient_phone": appointment.patient_phone,
                        "patient_email": appointment.patient_email,
                        "doctor_name": doctor.name,
                        "specialty": doctor.specialty,
                        "hospital": doctor.hospital,
                        "appointment_date": appointment.appointment_date,
                        "appointment_time": appointment.appointment_time,
                        "total_cost": appointment.total_cost,
                        "currency": appointment.currency,
                        "status": appointment.status,
                        "doctor_contact": doctor.contact
                    }
                    
                    pdf_buffer = generate_pdf_receipt(booking_details)
                    if pdf_buffer:
                        st.download_button(
                            "📄 PDF Receipt",
                            pdf_buffer.getvalue(),
                            f"receipt_{appointment.booking_id}.pdf",
                            mime="application/pdf",
                            key=f"dl_pdf_{appointment.booking_id}"
                        )
                    else:
                        st.button("📄 PDF Error", disabled=True, key=f"pdf_err_{appointment.booking_id}")
                else:
                    st.button("📄 PDF N/A", disabled=True, key=f"pdf_na_{appointment.booking_id}")

def analytics_dashboard():
    """Enhanced analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    if 'appointments' not in st.session_state or not st.session_state.appointments:
        st.info("📈 No data available for analytics")
        st.markdown("Book some appointments to see detailed analytics!")
        if st.button("📅 Book Appointment", use_container_width=True):
            st.rerun()
        return
    
    appointments = st.session_state.appointments
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_appointments = len(appointments)
    active_appointments = sum(1 for apt in appointments if apt.status == "Confirmed")
    cancelled_appointments = sum(1 for apt in appointments if apt.status == "Cancelled")
    total_spent = sum(apt.total_cost for apt in appointments if apt.status != "Cancelled")
    
    with col1:
        st.metric("📅 Total Appointments", total_appointments)
    with col2:
        st.metric("✅ Active", active_appointments)
    with col3:
        st.metric("❌ Cancelled", cancelled_appointments)
    with col4:
        st.metric("💰 Total Spent", f"{total_spent:.0f}")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        # Appointments by specialty
        st.subheader("📊 Appointments by Specialty")
        specialty_data = {}
        for apt in appointments:
            doctor = next((d for d in DOCTORS_DATABASE if d.id == apt.doctor_id), None)
            if doctor:
                specialty = doctor.specialty
                specialty_data[specialty] = specialty_data.get(specialty, 0) + 1
        
        if specialty_data:
            specialty_df = pd.DataFrame(list(specialty_data.items()), columns=['Specialty', 'Count'])
            fig = px.pie(specialty_df, values='Count', names='Specialty', 
                        title="Distribution by Medical Specialty")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Appointments by status
        st.subheader("📈 Appointment Status")
        status_data = {}
        for apt in appointments:
            status_data[apt.status] = status_data.get(apt.status, 0) + 1
        
        if status_data:
            status_df = pd.DataFrame(list(status_data.items()), columns=['Status', 'Count'])
            fig = px.bar(status_df, x='Status', y='Count', 
                        title="Appointments by Status",
                        color='Status')
            st.plotly_chart(fig, use_container_width=True)

def export_interface():
    """Enhanced data export interface"""
    st.header("📄 Export Data")
    
    if 'appointments' not in st.session_state or not st.session_state.appointments:
        st.info("📭 No appointment data to export")
        st.markdown("Book some appointments first to export data!")
        if st.button("📅 Book Appointment", use_container_width=True):
            st.rerun()
        return
    
    appointments = st.session_state.appointments
    
    st.markdown(f"**Available Data:** {len(appointments)} appointments ready for export")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 CSV Export")
        st.markdown("Export as CSV for Excel, Google Sheets, or data analysis")
        
        if st.button("📊 Generate CSV", use_container_width=True):
            csv_data = generate_csv_export()
            if csv_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="⬇️ Download CSV File",
                    data=csv_data,
                    file_name=f"appointments_export_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success("✅ CSV file generated successfully!")
    
    with col2:
        st.subheader("📋 JSON Export")
        st.markdown("Export as JSON for developers or backup purposes")
        
        if st.button("📋 Generate JSON", use_container_width=True):
            json_data = generate_json_export()
            if json_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="⬇️ Download JSON File",
                    data=json_data,
                    file_name=f"appointments_export_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
                st.success("✅ JSON file generated successfully!")

def generate_csv_export():
    """Generate CSV export data with consistent doctor information"""
    if 'appointments' not in st.session_state:
        return None
    
    appointments_data = []
    for appointment in st.session_state.appointments:
        # FIXED: Find doctor by exact ID match
        doctor = None
        for d in DOCTORS_DATABASE:
            if d.id == appointment.doctor_id:
                doctor = d
                break
        
        appointments_data.append({
            'Booking_ID': appointment.booking_id,
            'Patient_Name': appointment.patient_name,
            'Patient_Phone': appointment.patient_phone,
            'Patient_Email': appointment.patient_email,
            'Doctor_ID': appointment.doctor_id,
            'Doctor_Name': doctor.name if doctor else 'Unknown',
            'Doctor_Specialty': doctor.specialty if doctor else 'N/A',
            'Hospital': doctor.hospital if doctor else 'N/A',
            'Doctor_Contact': doctor.contact if doctor else 'N/A',
            'Doctor_Rating': doctor.rating if doctor else 'N/A',
            'Doctor_Experience': doctor.experience if doctor else 'N/A',
            'Appointment_Date': appointment.appointment_date,
            'Appointment_Time': appointment.appointment_time,
            'Total_Cost': appointment.total_cost,
            'Currency': appointment.currency,
            'Status': appointment.status,
            'Booking_Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    df = pd.DataFrame(appointments_data)
    return df.to_csv(index=False)

def generate_json_export():
    """Generate JSON export data with consistent doctor information"""
    if 'appointments' not in st.session_state:
        return None
    
    export_data = {
        'export_info': {
            'generated_at': datetime.now().isoformat(),
            'total_appointments': len(st.session_state.appointments),
            'version': '1.0'
        },
        'appointments': []
    }
    
    for appointment in st.session_state.appointments:
        # FIXED: Find doctor by exact ID match
        doctor = None
        for d in DOCTORS_DATABASE:
            if d.id == appointment.doctor_id:
                doctor = d
                break
        
        appointment_data = {
            'booking_id': appointment.booking_id,
            'patient_info': {
                'name': appointment.patient_name,
                'phone': appointment.patient_phone,
                'email': appointment.patient_email
            },
            'doctor_info': {
                'id': appointment.doctor_id,
                'name': doctor.name if doctor else 'Unknown',
                'specialty': doctor.specialty if doctor else 'N/A',
                'hospital': doctor.hospital if doctor else 'N/A',
                'contact': doctor.contact if doctor else 'N/A',
                'rating': doctor.rating if doctor else None,
                'experience': doctor.experience if doctor else None
            },
            'appointment_details': {
                'date': appointment.appointment_date,
                'time': appointment.appointment_time,
                'cost': appointment.total_cost,
                'currency': appointment.currency,
                'status': appointment.status
            }
        }
        
        export_data['appointments'].append(appointment_data)
    
    return json.dumps(export_data, indent=2)

# ========================================================================================
# MAIN APPLICATION - FIXED NAVIGATION
# ========================================================================================

def main():
    """FIXED: Enhanced main application with proper navigation and error handling"""
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'search'
    if 'llm' not in st.session_state:
        with st.spinner("🤖 Loading AI model..."):
            st.session_state.llm = load_open_source_llm()
    
    # App header
    st.title("🏥 AI-Powered Doctor Finder & Booking System")
    st.markdown("**Find and book appointments with 60+ doctors using advanced AI agents**")
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("🧭 Navigation")
        
        # Main navigation buttons with proper state management
        nav_buttons = [
            ("🔬 Doctor Search", "search"),
            ("📅 My Appointments", "appointments"),
            ("📊 Analytics", "analytics"),
            ("📄 Export Data", "export")
        ]
        
        for label, page in nav_buttons:
            button_type = "primary" if st.session_state.current_page == page else "secondary"
            if st.button(label, use_container_width=True, type=button_type, key=f"nav_{page}"):
                st.session_state.current_page = page
                # Clear booking state when navigating
                if 'selected_doctor_for_booking' in st.session_state:
                    del st.session_state.selected_doctor_for_booking
                if 'booking_step' in st.session_state:
                    del st.session_state.booking_step
                if 'last_booking' in st.session_state:
                    del st.session_state.last_booking
                st.rerun()
        
        st.divider()
        
        # Current page indicator
        page_names = {
            'search': '🔬 Doctor Search',
            'appointments': '📅 My Appointments',
            'analytics': '📊 Analytics Dashboard',
            'export': '📄 Export Data'
        }
        current_page_name = page_names.get(st.session_state.current_page, 'Unknown')
        st.markdown(f"**Current Page:** {current_page_name}")
        
        # Quick stats
        if 'appointments' in st.session_state and st.session_state.appointments:
            st.divider()
            st.markdown("### 📊 Quick Stats")
            total = len(st.session_state.appointments)
            active = sum(1 for apt in st.session_state.appointments if apt.status == "Confirmed")
            st.metric("Total Appointments", total)
            st.metric("Active Appointments", active)
        
        st.divider()
        
        # System info
        st.markdown("### ℹ️ System Info")
        st.markdown(f"**Doctors Available:** {len(DOCTORS_DATABASE)}")
        st.markdown(f"**Locations:** {len(set(doc.location for doc in DOCTORS_DATABASE))}")
        st.markdown(f"**Specialties:** {len(set(doc.specialty for doc in DOCTORS_DATABASE))}")
        
        st.divider()
        
        # Features
        st.markdown("### ✨ Features")
        st.markdown("""
        - 🔬 **Advanced Search** with filters
        - 👨‍⚕️ **60+ Doctors** across Pakistan
        - 📅 **Smart Booking** system
        - 📊 **Analytics** dashboard
        - 📄 **Data Export** (CSV/JSON)
        - 📱 **Mobile Responsive**
        - 📄 **PDF Receipts**
        """)
        
        # FIXED: Add system status
        st.divider()
        st.markdown("### 🔧 System Status")
        st.success("✅ All systems operational")
        if PDF_AVAILABLE:
            st.success("✅ PDF generation available")
        else:
            st.warning("⚠️ PDF generation unavailable")
    
    # FIXED: Route to appropriate interface with comprehensive error handling
    try:
        if st.session_state.current_page == 'search':
            advanced_search_interface()
            
            # Show search results if available
            if 'search_results' in st.session_state and st.session_state.search_results:
                st.markdown("---")
                display_search_results()
            
            # Show booking form if doctor selected
            if 'selected_doctor_for_booking' in st.session_state:
                show_patient_booking_form(st.session_state.selected_doctor_for_booking)
                
        elif st.session_state.current_page == 'appointments':
            appointments_interface()
        elif st.session_state.current_page == 'analytics':
            analytics_dashboard()
        elif st.session_state.current_page == 'export':
            export_interface()
        else:
            # Default fallback
            st.session_state.current_page = 'search'
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.error("Please try refreshing the page or navigate to a different section.")
        
        # Error recovery options
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Reset Application"):
                # Clear problematic session state
                for key in list(st.session_state.keys()):
                    if key not in ['llm']:  # Keep the LLM loaded
                        del st.session_state[key]
                st.session_state.current_page = 'search'
                st.rerun()
        
        with col2:
            if st.button("🏠 Go to Home"):
                st.session_state.current_page = 'search'
                if 'selected_doctor_for_booking' in st.session_state:
                    del st.session_state.selected_doctor_for_booking
                if 'booking_step' in st.session_state:
                    del st.session_state.booking_step
                st.rerun()
        
        with col3:
            if st.button("🔄 Refresh Page"):
                st.rerun()
        
        # Debug information (only in development)
        with st.expander("🔧 Debug Information", expanded=False):
            st.code(f"""
            Current Page: {st.session_state.current_page}
            Session Keys: {list(st.session_state.keys())}
            Error: {str(e)}
            Database Size: {len(DOCTORS_DATABASE)}
            PDF Available: {PDF_AVAILABLE}
            """)

# ========================================================================================
# RUN THE APPLICATION
# ========================================================================================

if __name__ == "__main__":
    main()
