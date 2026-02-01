"""
Клас для відстеження статусу задачі генерації
"""
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class GenerationTask:
    """Задача генерації 3D моделі"""
    task_id: str
    request: object
    status: str = "pending"  # pending, processing, completed, failed
    progress: int = 0  # 0-100
    message: str = ""
    # Основний файл, який повертається за замовчуванням (наприклад, 3MF або STL)
    output_file: Optional[str] = None
    # Набір доступних файлів по форматах: {"3mf": "...", "stl": "..."}
    output_files: Dict[str, str] = field(default_factory=dict)
    # Набір хмарних посилань: {"base_stl": "...", "3mf": "..."}
    firebase_outputs: Dict[str, str] = field(default_factory=dict)
    firebase_url: Optional[str] = None
    error: Optional[str] = None
    
    def update_status(self, status: str, progress: int, message: str = ""):
        """Оновлює статус задачі"""
        self.status = status
        self.progress = progress
        self.message = message
    
    def complete(self, output_file: str):
        """Позначає задачу як виконану"""
        self.status = "completed"
        self.progress = 100
        self.output_file = output_file

    def set_output(self, fmt: str, path: str):
        """Зберігає шлях до вихідного файлу для конкретного формату"""
        self.output_files[fmt.lower()] = path
    
    def fail(self, error: str):
        """Позначає задачу як невдалу"""
        self.status = "failed"
        self.error = error
        self.message = f"Помилка: {error}"

