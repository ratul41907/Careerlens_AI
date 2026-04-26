"""
Learning Pathway Generator - Efficient Single LLM Call
Generates personalized learning roadmaps using Ollama LLM
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import re


class LearningPathwayGenerator:
    """
    Generate personalized learning pathways using LLM (single efficient call)
    """

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "gemma:latest"

    def _call_ollama(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call Ollama LLM API with a single call"""
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
                timeout=90
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""

    def _build_fallback_pathway(self, skill_gaps: List[str], num_days: int) -> List[Dict]:
        """Build a complete pathway without LLM using smart templates"""
        focus_skills = skill_gaps[:4] if len(skill_gaps) >= 4 else skill_gaps

        # Phase labels based on day progression
        phases = ["Fundamentals", "Core Concepts", "Intermediate", "Advanced", "Projects", "Mastery"]

        resource_templates = {
            "YouTube Playlists 🎥": [
                "{skill} Full Course for Beginners - freeCodeCamp",
                "{skill} Tutorial - Traversy Media",
                "{skill} Crash Course - Fireship"
            ],
            "Online Courses 📚": [
                "{skill} Complete Bootcamp - Udemy",
                "{skill} Specialization - Coursera"
            ],
            "Documentation 📖": [
                "Official {skill} Documentation",
                "{skill} Getting Started Guide - Dev.to"
            ],
            "Practice Projects 💻": [
                "Build a {skill} CRUD application",
                "Create a {skill} REST API project"
            ],
            "Books & Articles 📕": [
                "Learning {skill} - O'Reilly Media",
                "{skill} Best Practices - Medium"
            ]
        }

        task_templates = [
            ["Watch 2 hours of {skill} fundamentals tutorial",
             "Read official {skill} documentation overview",
             "Set up {skill} development environment",
             "Complete beginner {skill} exercises"],
            ["Build a simple {skill} project from scratch",
             "Study {skill} core concepts and patterns",
             "Practice {skill} with online exercises",
             "Review {skill} best practices"],
            ["Implement {skill} in a real-world scenario",
             "Study advanced {skill} features",
             "Contribute to a {skill} open-source project",
             "Write a blog post about {skill} learnings"],
        ]

        daily_plans = []
        for day in range(1, num_days + 1):
            skill_idx = (day - 1) % len(focus_skills)
            current_skill = focus_skills[skill_idx]

            # Phase based on position in pathway
            phase_idx = min(int((day / num_days) * len(phases)), len(phases) - 1)
            phase = phases[phase_idx]

            # Task template based on phase
            task_set_idx = min(int((day / num_days) * len(task_templates)), len(task_templates) - 1)
            tasks = [t.format(skill=current_skill) for t in task_templates[task_set_idx]]

            # Resources
            resources = {}
            for res_type, templates in resource_templates.items():
                resources[res_type] = [t.format(skill=current_skill) for t in templates]

            # Time estimate increases with complexity
            if day <= num_days * 0.33:
                time_estimate = "3-4 hours"
                mini_project = f"Build a simple {current_skill} 'Hello World' application"
            elif day <= num_days * 0.66:
                time_estimate = "4-5 hours"
                mini_project = f"Create a {current_skill} CRUD application with tests"
            else:
                time_estimate = "5-6 hours"
                mini_project = f"Deploy a production-ready {current_skill} application"

            daily_plans.append({
                "day": day,
                "focus": f"Day {day}: {current_skill} — {phase}",
                "goal": f"Master {current_skill} {phase.lower()} concepts",
                "tasks": tasks,
                "resources": resources,
                "mini_project": mini_project,
                "time_estimate": time_estimate
            })

        return daily_plans

    def _parse_llm_pathway(self, llm_response: str, focus_skills: List[str], num_days: int) -> List[Dict]:
        """Parse LLM response into daily plans, fall back gracefully"""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Validate and fill structure
                    daily_plans = []
                    for i, item in enumerate(parsed[:num_days]):
                        day_num = i + 1
                        skill = focus_skills[(i) % len(focus_skills)]
                        daily_plans.append({
                            "day": day_num,
                            "focus": item.get("focus", f"Day {day_num}: {skill}"),
                            "goal": item.get("goal", f"Master {skill} concepts"),
                            "tasks": item.get("tasks", [f"Study {skill} for today"]),
                            "resources": item.get("resources", {
                                "YouTube Playlists 🎥": [f"{skill} Tutorial - freeCodeCamp"],
                                "Online Courses 📚": [f"{skill} Course - Udemy"]
                            }),
                            "mini_project": item.get("mini_project", f"Build a {skill} project"),
                            "time_estimate": item.get("time_estimate", "3-4 hours")
                        })
                    # Pad remaining days with fallback if LLM gave fewer
                    if len(daily_plans) < num_days:
                        fallback = self._build_fallback_pathway(focus_skills, num_days)
                        daily_plans.extend(fallback[len(daily_plans):])
                    return daily_plans
        except Exception:
            pass
        return None

    def generate_pathway(
        self,
        skill_gaps: List[str],
        jd_data: Dict,
        num_days: int = 7
    ) -> Dict:
        """
        Generate a complete learning pathway using a single LLM call

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

        # Determine focus skills
        if num_days == 7:
            focus_skills = skill_gaps[:2]
        elif num_days == 14:
            focus_skills = skill_gaps[:3]
        else:
            focus_skills = skill_gaps[:4]

        # Single LLM call for a sample of days (days 1, mid, last)
        # then use template logic to fill in the rest
        sample_days = min(3, num_days)  # Ask LLM for 3 representative days only
        skills_str = ", ".join(focus_skills)

        prompt = f"""Create a {num_days}-day learning plan for: {skills_str}

Return ONLY a JSON array with {sample_days} representative daily plans.
Each plan: {{"day": 1, "focus": "Day 1: SkillName - Fundamentals", "goal": "Learn basics", "tasks": ["task1", "task2", "task3"], "resources": {{"YouTube 🎥": ["Tutorial name - Channel"], "Courses 📚": ["Course - Platform"]}}, "mini_project": "Build X", "time_estimate": "3-4 hours"}}

JSON array only, no explanation:"""

        llm_response = self._call_ollama(prompt, max_tokens=1500)

        # Try to use LLM response, fall back to templates
        if llm_response and len(llm_response) > 50:
            parsed_plans = self._parse_llm_pathway(llm_response, focus_skills, sample_days)
        else:
            parsed_plans = None

        # Always build full pathway from templates (reliable)
        # Overlay any good LLM content for first few days
        full_pathway = self._build_fallback_pathway(focus_skills, num_days)

        if parsed_plans:
            # Overlay LLM-generated days onto the template pathway
            for llm_day in parsed_plans:
                day_idx = llm_day.get("day", 1) - 1
                if 0 <= day_idx < len(full_pathway):
                    full_pathway[day_idx].update({
                        "focus": llm_day.get("focus", full_pathway[day_idx]["focus"]),
                        "goal": llm_day.get("goal", full_pathway[day_idx]["goal"]),
                        "tasks": llm_day.get("tasks", full_pathway[day_idx]["tasks"]),
                        "mini_project": llm_day.get("mini_project", full_pathway[day_idx]["mini_project"]),
                    })

        completion_date = datetime.now() + timedelta(days=num_days)

        return {
            "success": True,
            "timeline_days": num_days,
            "focus_skills": focus_skills,
            "daily_plans": full_pathway,
            "estimated_daily_hours": "3-6 hours (varies by day)",
            "completion_date": completion_date.strftime("%B %d, %Y"),
            "total_estimated_hours": f"{num_days * 4}-{num_days * 6} hours"
        }


if __name__ == "__main__":
    generator = LearningPathwayGenerator()
    pathway = generator.generate_pathway(
        skill_gaps=["Docker", "Kubernetes"],
        jd_data={"text": "Need Docker and Kubernetes"},
        num_days=7
    )
    if pathway["success"]:
        print(f"✅ Generated {pathway['timeline_days']}-day pathway")
        print(f"Focus: {pathway['focus_skills']}")
        print(f"Day 1: {pathway['daily_plans'][0]['focus']}")