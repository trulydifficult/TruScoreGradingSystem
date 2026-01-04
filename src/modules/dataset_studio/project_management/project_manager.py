#!/usr/bin/env python3
"""
TruScore - Project Management System
Handles project discovery, creation, loading, and resource management
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import shutil

class ProjectManager:
    """
    Manages card grading projects with lazy loading and proper persistence
    """
    
    def __init__(self, base_projects_dir: str = "projects"):
        """
        Initialize project manager
        
        Args:
            base_projects_dir: Base directory for storing projects
        """
        self.base_dir = Path(base_projects_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Project structure
        self.current_project = None
        self.projects_cache = {}
    
    def discover_existing_projects(self) -> List[Dict]:
        """
        Scan for existing projects and return metadata
        
        Returns:
            List of project dictionaries with metadata
        """
        projects = []
        
        try:
            # Look for project directories
            for project_dir in self.base_dir.iterdir():
                if project_dir.is_dir():
                    project_info = self._load_project_metadata(project_dir)
                    if project_info:
                        projects.append(project_info)
            
            # Sort by last modified date (newest first)
            projects.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
            
            print(f"🔍 Discovered {len(projects)} existing projects")
            return projects
            
        except Exception as e:
            print(f"❌ Error discovering projects: {e}")
            return []
    
    def _load_project_metadata(self, project_dir: Path) -> Optional[Dict]:
        """
        Load project metadata from directory
        
        Args:
            project_dir: Path to project directory
            
        Returns:
            Project metadata dictionary or None
        """
        try:
            metadata_file = project_dir / "project_metadata.json"
            
            if metadata_file.exists():
                # Load existing metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                # Create metadata from directory contents
                metadata = self._generate_metadata_from_contents(project_dir)
                if metadata:
                    self._save_project_metadata(project_dir, metadata)
            
            # Add computed fields
            if metadata:
                metadata['project_path'] = str(project_dir)
                metadata['project_name'] = project_dir.name
                
            return metadata
            
        except Exception as e:
            print(f"⚠️ Error loading metadata for {project_dir.name}: {e}")
            return None
    
    def _generate_metadata_from_contents(self, project_dir: Path) -> Optional[Dict]:
        """
        Generate metadata by scanning project directory contents
        
        Args:
            project_dir: Path to project directory
            
        Returns:
            Generated metadata dictionary
        """
        try:
            # Look for common dataset files
            images_count = 0
            labels_count = 0
            annotations_count = 0
            
            # Check for images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            for ext in image_extensions:
                images_count += len(list(project_dir.glob(f"**/*{ext}")))
                images_count += len(list(project_dir.glob(f"**/*{ext.upper()}")))
            
            # Check for labels
            labels_count = len(list(project_dir.glob("**/*.txt")))
            
            # Check for COCO annotations
            coco_files = list(project_dir.glob("**/*.json"))
            for coco_file in coco_files:
                if coco_file.name != "project_metadata.json":
                    try:
                        with open(coco_file, 'r') as f:
                            data = json.load(f)
                            if 'annotations' in data:
                                annotations_count += len(data['annotations'])
                    except:
                        pass
            
            # Get directory stats
            stat = project_dir.stat()
            
            metadata = {
                'project_name': project_dir.name,
                'created_date': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'images_count': images_count,
                'labels_count': labels_count,
                'annotations_count': annotations_count,
                'project_type': 'card_grading',
                'status': 'discovered',
                'description': f"Auto-discovered project with {images_count} images"
            }
            
            return metadata if images_count > 0 or labels_count > 0 else None
            
        except Exception as e:
            print(f"❌ Error generating metadata for {project_dir.name}: {e}")
            return None
    
    def create_new_project(self, project_name: str, description: str = "", dataset_type: str = "card_grading") -> Dict:
        """
        Create a new project with proper structure
        
        Args:
            project_name: Name of the new project
            description: Optional project description
            
        Returns:
            Project metadata dictionary
        """
        try:
            # Sanitize project name
            safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            
            if not safe_name:
                safe_name = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create project directory
            project_dir = self.base_dir / safe_name
            
            # Handle name conflicts
            counter = 1
            original_name = safe_name
            while project_dir.exists():
                safe_name = f"{original_name}_{counter}"
                project_dir = self.base_dir / safe_name
                counter += 1
            
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Create project structure
            (project_dir / "images").mkdir(exist_ok=True)
            (project_dir / "labels").mkdir(exist_ok=True)
            (project_dir / "annotations").mkdir(exist_ok=True)
            (project_dir / "exports").mkdir(exist_ok=True)
            (project_dir / "models").mkdir(exist_ok=True)
            
            # Create metadata
            metadata = {
                'project_name': safe_name,
                'display_name': project_name,
                'description': description,
                'created_date': datetime.now().isoformat(),
                'last_modified': datetime.now().isoformat(),
                'images_count': 0,
                'labels_count': 0,
                'annotations_count': 0,
                'project_type': dataset_type,  # Dynamic dataset type from dialog!
                'status': 'created',
                'version': '1.0'
            }
            
            # Save metadata
            self._save_project_metadata(project_dir, metadata)
            
            # Add computed fields
            metadata['project_path'] = str(project_dir)
            
            print(f" Created new project: {safe_name}")
            return metadata
            
        except Exception as e:
            print(f"❌ Error creating project: {e}")
            raise
    
    def _save_project_metadata(self, project_dir: Path, metadata: Dict):
        """Save project metadata to file"""
        try:
            metadata_file = project_dir / "project_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")
    
    def load_project(self, project_path: str) -> Dict:
        """
        Load an existing project
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Project metadata
        """
        try:
            project_dir = Path(project_path)
            metadata = self._load_project_metadata(project_dir)
            
            if metadata:
                self.current_project = metadata
                return metadata
            else:
                raise ValueError(f"Could not load project from {project_path}")
                
        except Exception as e:
            print(f"❌ Error loading project: {e}")
            raise
    
    def delete_project(self, project_path: str, confirm: bool = False) -> bool:
        """
        Delete a project (with confirmation)
        
        Args:
            project_path: Path to project directory
            confirm: Confirmation flag
            
        Returns:
            True if deleted successfully
        """
        try:
            if not confirm:
                return False
            
            project_dir = Path(project_path)
            if project_dir.exists():
                shutil.rmtree(project_dir)
                print(f"🗑️ Deleted project: {project_dir.name}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error deleting project: {e}")
            return False
    
    def update_project_stats(self, project_path: str, images_count: int = None, 
                           labels_count: int = None, annotations_count: int = None):
        """Update project statistics"""
        try:
            project_dir = Path(project_path)
            metadata = self._load_project_metadata(project_dir)
            
            if metadata:
                if images_count is not None:
                    metadata['images_count'] = images_count
                if labels_count is not None:
                    metadata['labels_count'] = labels_count
                if annotations_count is not None:
                    metadata['annotations_count'] = annotations_count
                
                metadata['last_modified'] = datetime.now().isoformat()
                self._save_project_metadata(project_dir, metadata)
                
        except Exception as e:
            print(f"❌ Error updating project stats: {e}")
    
    def get_project_paths(self, project_path: str) -> Dict[str, Path]:
        """
        Get standard paths for a project
        
        Args:
            project_path: Path to project directory
            
        Returns:
            Dictionary of standard paths
        """
        project_dir = Path(project_path)
        
        return {
            'project': project_dir,
            'images': project_dir / "images",
            'labels': project_dir / "labels", 
            'annotations': project_dir / "annotations",
            'exports': project_dir / "exports",
            'models': project_dir / "models"
        }
    
    def export_project(self, project_path: str, export_path: str) -> bool:
        """Export project to a zip file"""
        try:
            import zipfile
            
            project_dir = Path(project_path)
            export_file = Path(export_path)
            
            with zipfile.ZipFile(export_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in project_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(project_dir.parent)
                        zipf.write(file_path, arcname)
            
            print(f"📤 Exported project to: {export_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting project: {e}")
            return False


def main():
    """Test the project manager"""
    pm = ProjectManager()
    
    # Discover existing projects
    projects = pm.discover_existing_projects()
    print(f"Found {len(projects)} projects:")
    for project in projects:
        print(f"  - {project['project_name']}: {project['images_count']} images")
    
    # Create a test project
    new_project = pm.create_new_project("Test Project", "Testing project management")
    print(f"Created: {new_project}")


if __name__ == "__main__":
    main()