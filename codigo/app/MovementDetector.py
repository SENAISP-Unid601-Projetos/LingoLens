import numpy as np
from collections import deque
import logging

class MovementDetector:
    def __init__(self, sequence_length=10, threshold=0.15):
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.landmark_buffer = deque(maxlen=sequence_length)
        self.is_moving = False
        self.movement_counter = 0
        
    def add_landmarks(self, landmarks):
        """Adiciona landmarks ao buffer e detecta movimento"""
        if landmarks is None:
            return False
            
        self.landmark_buffer.append(landmarks)
        
        # Precisa de sequência mínima para detectar movimento
        if len(self.landmark_buffer) < 2:
            return False
            
        return self._calculate_movement()
    
    def _calculate_movement(self):
        """Calcula se há movimento significativo"""
        try:
            # Calcular diferença entre frames consecutivos
            total_movement = 0
            frame_count = 0
            
            for i in range(1, len(self.landmark_buffer)):
                current = np.array(self.landmark_buffer[i])
                previous = np.array(self.landmark_buffer[i-1])
                
                if current.shape == previous.shape:
                    movement = np.mean(np.abs(current - previous))
                    total_movement += movement
                    frame_count += 1
            
            if frame_count > 0:
                avg_movement = total_movement / frame_count
                self.is_moving = avg_movement > self.threshold
                
                # Contador para evitar flickering
                if self.is_moving:
                    self.movement_counter = min(self.movement_counter + 1, self.sequence_length)
                else:
                    self.movement_counter = max(self.movement_counter - 1, 0)
                
                # Só considera movimento se persistir por vários frames
                return self.movement_counter > (self.sequence_length // 3)
            
            return False
            
        except Exception as e:
            logging.error(f"❌ Erro ao calcular movimento: {e}")
            return False
    
    def reset(self):
        """Reseta o detector de movimento"""
        self.landmark_buffer.clear()
        self.is_moving = False
        self.movement_counter = 0
    
    def get_movement_intensity(self):
        """Retorna a intensidade do movimento (0-1)"""
        return min(self.movement_counter / self.sequence_length, 1.0)