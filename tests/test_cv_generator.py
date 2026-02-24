"""
Test CV Generator
"""
from src.generation.cv_generator import CVGenerator
import os


def test_cv_generator():
    """Test CV generation with sample data"""
    
    print("=" * 70)
    print("CV GENERATOR TEST")
    print("=" * 70)
    
    # Sample data
    personal_info = {
        'name': 'John Doe',
        'email': 'john.doe@email.com',
        'phone': '+1-234-567-8900',
        'location': 'San Francisco, CA',
        'linkedin': 'linkedin.com/in/johndoe',
        'summary': 'Results-driven Software Engineer with 5+ years of experience building scalable web applications. Expertise in Python, FastAPI, and cloud technologies. Proven track record of delivering high-quality solutions that improve system performance by 40%.'
    }
    
    experience = [
        {
            'title': 'Senior Software Engineer',
            'company': 'Tech Corp',
            'duration': 'January 2021 - Present',
            'bullets': [
                'Developed REST APIs using FastAPI, serving 100K+ daily requests with 99.9% uptime',
                'Architected microservices deployed on AWS using Docker and Kubernetes, reducing infrastructure costs by 30%',
                'Improved system performance by 40% through database optimization and caching strategies',
                'Mentored team of 3 junior developers, conducting code reviews and technical guidance'
            ]
        },
        {
            'title': 'Software Engineer',
            'company': 'StartupXYZ',
            'duration': 'June 2019 - December 2020',
            'bullets': [
                'Built React-based dashboard for data visualization, used by 5000+ users',
                'Implemented CI/CD pipelines with GitHub Actions, reducing deployment time by 50%',
                'Integrated third-party APIs (Stripe, SendGrid) for payment and email functionality',
                'Collaborated with cross-functional teams using Agile methodology'
            ]
        }
    ]
    
    education = [
        {
            'degree': 'Bachelor of Science in Computer Science',
            'institution': 'University of Technology',
            'year': '2015 - 2019',
            'gpa': '3.8/4.0'
        }
    ]
    
    skills = [
        'Python', 'JavaScript', 'TypeScript',
        'FastAPI', 'Django', 'React', 'Node.js',
        'PostgreSQL', 'MongoDB', 'Redis',
        'Docker', 'Kubernetes', 'AWS', 'Git',
        'CI/CD', 'Agile', 'REST APIs'
    ]
    
    projects = [
        {
            'name': 'E-Commerce Platform',
            'description': 'Built a full-stack e-commerce platform serving 10K+ users with real-time inventory management and payment processing.',
            'technologies': 'FastAPI, React, PostgreSQL, Redis, Stripe API'
        },
        {
            'name': 'ML Recommendation Engine',
            'description': 'Developed machine learning model for product recommendations, increasing conversion rate by 25%.',
            'technologies': 'Python, TensorFlow, scikit-learn, FastAPI'
        }
    ]
    
    certifications = [
        'AWS Certified Solutions Architect - Associate (2022)',
        'Google Cloud Professional Cloud Developer (2021)',
        'Certified Kubernetes Administrator (CKA) (2023)'
    ]
    
    print("\n📝 GENERATING CV...")
    print("-" * 70)
    
    # Generate CV
    generator = CVGenerator()
    doc = generator.generate_cv(
        personal_info=personal_info,
        experience=experience,
        education=education,
        skills=skills,
        projects=projects,
        certifications=certifications
    )
    
    # Save to file
    output_dir = "data/sample_cvs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_cv_john_doe.docx")
    
    generator.save_cv(doc, output_path)
    
    print(f"\n✅ CV Generated Successfully!")
    print(f"   Saved to: {output_path}")
    print(f"\n📊 CV Contents:")
    print(f"   Name: {personal_info['name']}")
    print(f"   Experience: {len(experience)} positions")
    print(f"   Education: {len(education)} degrees")
    print(f"   Skills: {len(skills)} skills listed")
    print(f"   Projects: {len(projects)} projects")
    print(f"   Certifications: {len(certifications)} certifications")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETED!")
    print("=" * 70)
    print(f"\n💡 Open the generated CV:")
    print(f"   {os.path.abspath(output_path)}")


if __name__ == "__main__":
    test_cv_generator()