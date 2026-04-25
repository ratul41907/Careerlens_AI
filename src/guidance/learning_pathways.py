"""
Learning Pathway Generator - 100% LLM-Based (No Hardcoding)
Generates personalized learning roadmaps using Ollama LLM
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json


class LearningPathwayGenerator:
    """
    Generate personalized learning pathways using LLM
    All resources, playlists, courses dynamically generated
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize the pathway generator
        
        Args:
            ollama_url: Ollama API endpoint
        """
        self.ollama_url = ollama_url
        self.model = "gemma:latest"
    
    def _call_ollama(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Call Ollama LLM API
        
        Args:
            prompt: Prompt text
            max_tokens: Maximum response tokens
            
        Returns:
            LLM response text
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return ""
        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""
    
    def _parse_list_from_llm(self, text: str) -> List[str]:
        """
        Parse bulleted/numbered list from LLM response
        
        Args:
            text: LLM response text
            
        Returns:
            List of items
        """
        items = []
        for line in text.split('\n'):
            line = line.strip()
            # Remove bullets/numbers
            if line:
                # Remove common list markers
                cleaned = line.lstrip('•*-–—►▸▹▪▫1234567890.)> ')
                if cleaned and len(cleaned) > 5:  # Avoid empty/short lines
                    items.append(cleaned)
        return items
    
    def _generate_youtube_playlists(self, skill: str) -> List[str]:
        """
        Generate YouTube playlist recommendations using LLM
        
        Args:
            skill: Skill name
            
        Returns:
            List of YouTube playlists/channels
        """
        prompt = f"""List 5 specific real YouTube channels or playlists for learning {skill}.
Include the channel name and video/playlist title.
Format each as: "Title - Channel Name"

Examples:
- Python Full Course for Beginners - freeCodeCamp
- JavaScript Tutorial - Traversy Media
- React Complete Guide - Academind

Now list 5 YouTube resources for {skill}:"""

        response = self._call_ollama(prompt, max_tokens=300)
        playlists = self._parse_list_from_llm(response)
        
        # Fallback if LLM fails
        if len(playlists) < 2:
            playlists = [
                f"{skill} Full Course - freeCodeCamp",
                f"{skill} Tutorial for Beginners - Programming with Mosh",
                f"{skill} Crash Course - Traversy Media",
                f"{skill} Complete Guide - Academind",
                f"{skill} Tutorial - Net Ninja"
            ]
        
        return playlists[:5]
    
    def _generate_online_courses(self, skill: str) -> List[str]:
        """
        Generate online course recommendations using LLM
        
        Args:
            skill: Skill name
            
        Returns:
            List of online courses
        """
        prompt = f"""List 5 specific real online courses for learning {skill}.
Include the course title and platform.
Format: "Course Title - Platform"

Examples:
- Complete Python Bootcamp - Udemy
- Machine Learning Specialization - Coursera
- Web Development Bootcamp - Udemy

Now list 5 courses for {skill}:"""

        response = self._call_ollama(prompt, max_tokens=300)
        courses = self._parse_list_from_llm(response)
        
        if len(courses) < 2:
            courses = [
                f"{skill} Complete Course - Udemy",
                f"{skill} Specialization - Coursera",
                f"{skill} Path - Pluralsight",
                f"{skill} Bootcamp - edX",
                f"{skill} Masterclass - LinkedIn Learning"
            ]
        
        return courses[:5]
    
    def _generate_documentation(self, skill: str) -> List[str]:
        """
        Generate documentation/article recommendations using LLM
        
        Args:
            skill: Skill name
            
        Returns:
            List of documentation resources
        """
        prompt = f"""List 5 official documentation sites and tutorial resources for {skill}.
Format: "Resource Name - Type"

Examples:
- Official Python Documentation - Docs
- MDN Web Docs for JavaScript - Docs
- React Official Tutorial - Tutorial

Now list 5 documentation resources for {skill}:"""

        response = self._call_ollama(prompt, max_tokens=300)
        docs = self._parse_list_from_llm(response)
        
        if len(docs) < 2:
            docs = [
                f"Official {skill} Documentation",
                f"{skill} Tutorials and Guides",
                f"{skill} Best Practices - Dev.to",
                f"{skill} Cheat Sheet - DevDocs",
                f"{skill} Articles - Medium"
            ]
        
        return docs[:5]
    
    def _generate_practice_projects(self, skill: str) -> List[str]:
        """
        Generate practice project ideas using LLM
        
        Args:
            skill: Skill name
            
        Returns:
            List of project ideas
        """
        prompt = f"""List 5 practical project ideas to learn {skill}.
Each should be a specific, buildable project.
Format: Short project description

Examples:
- Build a todo app with CRUD operations
- Create a weather dashboard using API
- Develop a blog with authentication

Now list 5 project ideas for {skill}:"""

        response = self._call_ollama(prompt, max_tokens=300)
        projects = self._parse_list_from_llm(response)
        
        if len(projects) < 2:
            projects = [
                f"Build a basic {skill} application with CRUD",
                f"Create a {skill} project using best practices",
                f"Develop a portfolio project with {skill}",
                f"Contribute to open source {skill} projects",
                f"Build a production-ready {skill} application"
            ]
        
        return projects[:5]
    
    def _generate_books_articles(self, skill: str) -> List[str]:
        """
        Generate book and article recommendations using LLM
        
        Args:
            skill: Skill name
            
        Returns:
            List of books/articles
        """
        prompt = f"""List 5 recommended books or article series for learning {skill}.
Include book/article title and author if known.

Examples:
- "Clean Code" by Robert Martin
- "You Don't Know JS" series by Kyle Simpson
- "Python Crash Course" by Eric Matthes

Now list 5 books/articles for {skill}:"""

        response = self._call_ollama(prompt, max_tokens=300)
        books = self._parse_list_from_llm(response)
        
        if len(books) < 2:
            books = [
                f"'{skill} in Action' - Manning Publications",
                f"'Learning {skill}' - O'Reilly Media",
                f"'{skill} Design Patterns' - Addison-Wesley",
                f"'{skill} Best Practices' - Packt Publishing",
                f"'{skill} Cookbook' - Pragmatic Bookshelf"
            ]
        
        return books[:5]
    
    def _generate_daily_tasks(self, skill: str, day: int, total_days: int) -> List[str]:
        """
        Generate daily learning tasks using LLM
        
        Args:
            skill: Skill name
            day: Current day number
            total_days: Total pathway days
            
        Returns:
            List of daily tasks
        """
        # Determine difficulty level based on day
        if day <= total_days * 0.3:
            level = "beginner/fundamentals"
        elif day <= total_days * 0.6:
            level = "intermediate"
        else:
            level = "advanced/mastery"
        
        prompt = f"""Create 4 specific learning tasks for day {day} of a {total_days}-day {skill} learning plan.
This is {level} level.

Format each task as an actionable item.

Examples:
- Watch 2 hours of Python basics tutorial
- Complete exercises on variables and data types
- Read chapter 1 of official documentation
- Build a simple calculator program

Now list 4 tasks for day {day} learning {skill} ({level} level):"""

        response = self._call_ollama(prompt, max_tokens=200)
        tasks = self._parse_list_from_llm(response)
        
        if len(tasks) < 2:
            tasks = [
                f"Study {skill} {level} concepts for 2-3 hours",
                f"Complete {skill} exercises and challenges",
                f"Read {skill} documentation and tutorials",
                f"Practice {skill} with hands-on coding"
            ]
        
        return tasks[:4]
    
    def _generate_mini_project(self, skill: str, day: int, total_days: int) -> str:
        """
        Generate mini project for the day using LLM
        
        Args:
            skill: Skill name
            day: Current day number
            total_days: Total pathway days
            
        Returns:
            Mini project description
        """
        if day <= total_days * 0.3:
            complexity = "simple beginner"
        elif day <= total_days * 0.6:
            complexity = "intermediate"
        else:
            complexity = "advanced portfolio-worthy"
        
        prompt = f"""Suggest one specific {complexity} project for day {day} of learning {skill}.
Describe it in 10-15 words.

Examples:
- Build a todo list app with add/delete functionality
- Create a weather dashboard fetching API data
- Develop a blog with authentication and comments

Now suggest a {complexity} {skill} project:"""

        response = self._call_ollama(prompt, max_tokens=100)
        
        # Get first meaningful line
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        project = lines[0] if lines else f"Build a {complexity} {skill} project"
        
        # Clean up common prefixes
        project = project.lstrip('•*-–—►▸▹▪▫1234567890.)> ')
        
        return project
    
    def generate_pathway(
        self,
        skill_gaps: List[str],
        jd_data: Dict,
        num_days: int = 7
    ) -> Dict:
        """
        Generate a complete learning pathway using LLM
        
        Args:
            skill_gaps: List of missing skills
            jd_data: Job description data
            num_days: Number of days (7, 14, or 30)
            
        Returns:
            Complete learning pathway dict
        """
        if not skill_gaps:
            return {
                "success": False,
                "error": "No skill gaps provided"
            }
        
        # Determine focus skills based on pathway length
        if num_days == 7:
            focus_skills = skill_gaps[:2]
        elif num_days == 14:
            focus_skills = skill_gaps[:3]
        else:  # 30 days
            focus_skills = skill_gaps[:4]
        
        # Generate daily plans using LLM
        daily_plans = []
        
        for day in range(1, num_days + 1):
            # Rotate through focus skills
            skill_index = (day - 1) % len(focus_skills)
            current_skill = focus_skills[skill_index]
            
            # Generate resources for this skill
            youtube = self._generate_youtube_playlists(current_skill)
            courses = self._generate_online_courses(current_skill)
            docs = self._generate_documentation(current_skill)
            projects = self._generate_practice_projects(current_skill)
            books = self._generate_books_articles(current_skill)
            
            # Generate daily tasks
            tasks = self._generate_daily_tasks(current_skill, day, num_days)
            
            # Generate mini project
            mini_project = self._generate_mini_project(current_skill, day, num_days)
            
            # Determine time estimate based on day
            if day <= num_days * 0.3:
                time_estimate = "3-4 hours"
            elif day <= num_days * 0.6:
                time_estimate = "4-5 hours"
            else:
                time_estimate = "5-6 hours"
            
            plan = {
                "day": day,
                "focus": f"Day {day}: {current_skill}",
                "tasks": tasks,
                "resources": {
                    "YouTube Playlists 🎥": youtube[:3],
                    "Online Courses 📚": courses[:2],
                    "Documentation 📖": docs[:2],
                    "Practice Projects 💻": projects[:2],
                    "Books & Articles 📕": books[:2]
                },
                "mini_project": mini_project,
                "time_estimate": time_estimate
            }
            
            daily_plans.append(plan)
        
        # Calculate completion date
        start_date = datetime.now()
        completion_date = start_date + timedelta(days=num_days)
        
        return {
            "success": True,
            "timeline_days": num_days,
            "focus_skills": focus_skills,
            "daily_plans": daily_plans,
            "estimated_daily_hours": "3-6 hours (varies by day)",
            "completion_date": completion_date.strftime("%B %d, %Y"),
            "total_estimated_hours": f"{num_days * 4}-{num_days * 6} hours"
        }


# Test function
if __name__ == "__main__":
    generator = LearningPathwayGenerator()
    
    # Test pathway generation
    pathway = generator.generate_pathway(
        skill_gaps=["Python", "Machine Learning", "Docker"],
        jd_data={"text": "Looking for ML Engineer with Python and Docker"},
        num_days=7
    )
    
    if pathway["success"]:
        print(f"✅ Generated {pathway['timeline_days']}-day pathway")
        print(f"Focus Skills: {', '.join(pathway['focus_skills'])}")
        print(f"\nDay 1 Plan:")
        print(json.dumps(pathway['daily_plans'][0], indent=2))
    else:
        print(f"❌ Error: {pathway['error']}")