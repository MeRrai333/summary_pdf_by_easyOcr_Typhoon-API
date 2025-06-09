import os
from pathlib import Path
import uuid
import mimetypes
from datetime import datetime
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UploadFileService:
    def __init__(self,  base_upload_dir: str = "./pdfSummaryProject/uploads"):
        self.base_upload_dir = Path(base_upload_dir)
        self.file_type_mappings = {
            """ 'video': {
                'extensions': ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.wmv', '.m4v', '.3gp'],
                'mime_types': ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 
                              'video/x-flv', 'video/webm', 'video/x-ms-wmv'],
                'directory': 'video'
            }, """
            'doc': {
                'extensions': ['.pdf'],
                'mime_types': ['application/pdf'],
                'directory': 'doc'
            },
        }

        logger.info(f"FileUploadService initialized with base directory: {self.base_upload_dir}")

    
    def get_file_type(self, filename: str, content_type: str) -> str:
        """
        Determine file type based on filename extension and MIME type.
        
        Args:
            filename (str): Name of the file
            content_type (str): MIME type of the file
            
        Returns:
            str: File type category
        """
        file_extension = Path(filename).suffix.lower()
        
        # Check each file type mapping
        for file_type, mapping in self.file_type_mappings.items():
            # Check by extension first
            if file_extension in mapping['extensions']:
                return file_type
            
            # Check by MIME type
            if content_type.lower() in mapping['mime_types']:
                return file_type
        
        return 'other'
    
    def generate_unique_filename(self, original_filename: str) -> str:
        """
        Generate a unique filename to prevent conflicts.
        
        Args:
            original_filename (str): Original filename
            
        Returns:
            str: Unique filename
        """
        # Get file extension
        file_path = Path(original_filename)
        name = file_path.stem
        extension = file_path.suffix
        
        # Generate unique identifier
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create unique filename: originalname_timestamp_uniqueid.ext
        unique_filename = f"{name}_{timestamp}_{unique_id}{extension}"
        
        return unique_filename
    
    def save_file(self, file_content: bytes, filename: str, content_type: str) -> Dict[str, Any]:
        """
        Save file to appropriate directory based on its type.
        
        Args:
            file_content (bytes): File content as bytes
            filename (str): Original filename
            content_type (str): MIME type
            
        Returns:
            Dict: File save information
        """
        try:
            # Determine file type and target directory
            file_type = self.get_file_type(filename, content_type)
            
            if file_type in self.file_type_mappings:
                target_dir = self.base_upload_dir / self.file_type_mappings[file_type]['directory']
            else:
                target_dir = self.base_upload_dir / 'other'
            
            # Generate unique filename
            unique_filename = self.generate_unique_filename(filename)
            file_path = target_dir / unique_filename
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Get file stats
            file_stats = file_path.stat()
            
            file_info = {
                'status': 'success',
                'original_filename': filename,
                'saved_filename': unique_filename,
                'file_type': file_type,
                'directory': str(target_dir.relative_to(self.base_upload_dir)),
                'full_path': str(file_path),
                'relative_path': str(file_path.relative_to(self.base_upload_dir)),
                'content_type': content_type,
                'size_bytes': len(file_content),
                'size_mb': round(len(file_content) / (1024 * 1024), 2),
                'created_at': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                'upload_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"File saved successfully: {filename} -> {file_path}")
            return file_info
            
        except Exception as e:
            logger.error(f"Error saving file {filename}: {e}")
            return {
                'status': 'error',
                'original_filename': filename,
                'error': str(e),
                'upload_timestamp': datetime.now().isoformat()
            }
    
    def save_multiple_files(self, files_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Save multiple files and organize them by type.
        
        Args:
            files_data (List[Dict]): List of file data dictionaries with keys:
                                   'content', 'filename', 'content_type'
        
        Returns:
            Dict: Summary of all file operations
        """
        results = {
            'successful_uploads': [],
            'failed_uploads': [],
            'summary_by_type': {},
            'total_files': len(files_data),
            'total_size_bytes': 0
        }
        
        for file_data in files_data:
            try:
                file_info = self.save_file(
                    file_content=file_data['content'],
                    filename=file_data['filename'],
                    content_type=file_data['content_type']
                )
                
                if file_info['status'] == 'success':
                    results['successful_uploads'].append(file_info)
                    results['total_size_bytes'] += file_info['size_bytes']
                    
                    # Update summary by type
                    file_type = file_info['file_type']
                    if file_type not in results['summary_by_type']:
                        results['summary_by_type'][file_type] = {
                            'count': 0,
                            'total_size_bytes': 0,
                            'files': []
                        }
                    
                    results['summary_by_type'][file_type]['count'] += 1
                    results['summary_by_type'][file_type]['total_size_bytes'] += file_info['size_bytes']
                    results['summary_by_type'][file_type]['files'].append({
                        'filename': file_info['saved_filename'],
                        'original_filename': file_info['original_filename'],
                        'size_mb': file_info['size_mb']
                    })
                else:
                    results['failed_uploads'].append(file_info)
                    
            except Exception as e:
                results['failed_uploads'].append({
                    'original_filename': file_data.get('filename', 'unknown'),
                    'status': 'error',
                    'error': str(e)
                })
        
        # Add summary statistics
        results['summary'] = {
            'successful_count': len(results['successful_uploads']),
            'failed_count': len(results['failed_uploads']),
            'success_rate': len(results['successful_uploads']) / len(files_data) * 100 if files_data else 0,
            'total_size_mb': round(results['total_size_bytes'] / (1024 * 1024), 2),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return results
    