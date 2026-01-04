"""
TruScore Professional - Premium Animated Loaders
=================================================
Next-level loading animations that make investors go "WOW".
Inspired by elite CSS animations but implemented in PyQt6 with smooth 60 FPS.

Loaders:
- OrbitingBallsLoader: Balls orbiting in a circle with fade
- DominoLoader: Cascading domino fall effect
- PulseRingLoader: Expanding rings with fade
- GlowPulseLoader: Pulsing glow orb
- BarWaveLoader: Wave effect with bars
- ParticleLoader: Particle explosion effect
"""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QTimer, QPointF, Qt, pyqtProperty, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPainter, QColor, QRadialGradient, QBrush, QPen
import math


class OrbitingBallsLoader(QWidget):
    """
    Orbiting balls with fade effect - sleek and professional.
    Perfect for loading operations.
    """
    
    def __init__(self, parent=None, size=80, num_balls=5, color=None):
        super().__init__(parent)
        
        self.loader_size = size
        self.num_balls = num_balls
        self.color = color or QColor(56, 189, 248)
        self.rotation = 0
        
        self.setFixedSize(size, size)
        
        # Animation timer - 60 FPS
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.setInterval(16)
    
    def start(self):
        """Start the loader animation"""
        self.timer.start()
        self.show()
    
    def stop(self):
        """Stop the loader animation"""
        self.timer.stop()
        self.hide()
    
    def animate(self):
        """Update animation"""
        self.rotation = (self.rotation + 3) % 360
        self.update()
    
    def paintEvent(self, event):
        """Draw orbiting balls"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center_x = self.width() / 2
        center_y = self.height() / 2
        orbit_radius = self.loader_size / 3
        ball_radius = self.loader_size / 12
        
        painter.translate(center_x, center_y)
        
        # Draw balls
        for i in range(self.num_balls):
            # Calculate position
            angle = (self.rotation + (i * 360 / self.num_balls)) * (math.pi / 180)
            x = math.cos(angle) * orbit_radius
            y = math.sin(angle) * orbit_radius
            
            # Calculate alpha based on position (fade effect)
            alpha = int(255 * (i / self.num_balls))
            
            # Create gradient for ball
            gradient = QRadialGradient(x, y, ball_radius)
            
            ball_color = QColor(self.color)
            ball_color.setAlpha(alpha)
            gradient.setColorAt(0, ball_color)
            
            ball_outer = QColor(self.color)
            ball_outer.setAlpha(0)
            gradient.setColorAt(1, ball_outer)
            
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x, y), ball_radius, ball_radius)
        
        painter.end()


class DominoLoader(QWidget):
    """
    Cascading domino effect - bars that fall in sequence.
    Unique and eye-catching.
    """
    
    def __init__(self, parent=None, size=100, num_bars=8, color=None):
        super().__init__(parent)
        
        self.loader_size = size
        self.num_bars = num_bars
        self.color = color or QColor(255, 255, 255)
        self.phase = 0
        
        self.setFixedSize(size, size)
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.setInterval(16)
    
    def start(self):
        self.timer.start()
        self.show()
    
    def stop(self):
        self.timer.stop()
        self.hide()
    
    def animate(self):
        self.phase = (self.phase + 2) % 360
        self.update()
    
    def paintEvent(self, event):
        """Draw domino bars"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        bar_width = self.loader_size / 2.5
        bar_height = 7
        spacing = 10
        
        start_x = (self.width() - (self.num_bars - 1) * spacing - bar_width) / 2
        center_y = self.height() / 2
        
        for i in range(self.num_bars):
            x = start_x + i * spacing
            
            # Calculate rotation based on phase
            delay = i * 45  # Delay between bars
            current_phase = (self.phase + delay) % 360
            
            # Rotation angle (0 to 90 degrees)
            if 0 <= current_phase < 180:
                t = current_phase / 180.0
                angle = t * 90
                opacity = 0.7 + (0.3 * math.sin(t * math.pi))
            else:
                angle = 0
                opacity = 1.0
            
            painter.save()
            painter.translate(x, center_y)
            painter.rotate(angle)
            
            # Draw bar with glow
            bar_color = QColor(self.color)
            bar_color.setAlpha(int(255 * opacity))
            
            # Shadow
            shadow_color = QColor(0, 0, 0, 100)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(shadow_color))
            painter.drawRect(int(-bar_width/2 + 2), int(-bar_height/2 + 2), int(bar_width), int(bar_height))
            
            # Bar
            painter.setBrush(QBrush(bar_color))
            painter.drawRect(int(-bar_width/2), int(-bar_height/2), int(bar_width), int(bar_height))
            
            painter.restore()
        
        painter.end()


class PulseRingLoader(QWidget):
    """
    Expanding rings that fade out - clean and modern.
    Perfect for background processes.
    """
    
    def __init__(self, parent=None, size=100, color=None):
        super().__init__(parent)
        
        self.loader_size = size
        self.color = color or QColor(56, 189, 248)
        self.rings = []
        self.max_rings = 3
        
        self.setFixedSize(size, size)
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.setInterval(16)
        
        # Ring spawn timer
        self.spawn_timer = QTimer(self)
        self.spawn_timer.timeout.connect(self.spawn_ring)
        self.spawn_timer.setInterval(600)
    
    def start(self):
        self.timer.start()
        self.spawn_timer.start()
        self.show()
    
    def stop(self):
        self.timer.stop()
        self.spawn_timer.stop()
        self.rings.clear()
        self.hide()
    
    def spawn_ring(self):
        """Create a new ring"""
        if len(self.rings) < self.max_rings:
            self.rings.append({'radius': 0, 'alpha': 1.0})
    
    def animate(self):
        """Update rings"""
        # Update existing rings
        for ring in self.rings[:]:
            ring['radius'] += 1.5
            ring['alpha'] -= 0.015
            
            # Remove if fully faded
            if ring['alpha'] <= 0:
                self.rings.remove(ring)
        
        self.update()
    
    def paintEvent(self, event):
        """Draw expanding rings"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center_x = self.width() / 2
        center_y = self.height() / 2
        max_radius = self.loader_size / 2
        
        for ring in self.rings:
            radius = ring['radius']
            alpha = ring['alpha']
            
            if radius < max_radius:
                ring_color = QColor(self.color)
                ring_color.setAlpha(int(255 * alpha))
                
                pen = QPen(ring_color)
                pen.setWidth(3)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                
                painter.drawEllipse(QPointF(center_x, center_y), radius, radius)
        
        painter.end()


class GlowPulseLoader(QWidget):
    """
    Pulsing glow orb - elegant and attention-grabbing.
    Great for important operations.
    """
    
    def __init__(self, parent=None, size=80, color=None):
        super().__init__(parent)
        
        self.loader_size = size
        self.color = color or QColor(168, 85, 247)  # Purple
        self.pulse_phase = 0
        
        self.setFixedSize(size, size)
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.setInterval(16)
    
    def start(self):
        self.timer.start()
        self.show()
    
    def stop(self):
        self.timer.stop()
        self.hide()
    
    def animate(self):
        self.pulse_phase = (self.pulse_phase + 3) % 360
        self.update()
    
    def paintEvent(self, event):
        """Draw pulsing glow"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center_x = self.width() / 2
        center_y = self.height() / 2
        
        # Calculate pulse
        pulse = (math.sin(self.pulse_phase * math.pi / 180) + 1) / 2  # 0 to 1
        base_radius = self.loader_size / 6
        radius = base_radius + (base_radius * 0.5 * pulse)
        glow_radius = radius * 2
        
        # Draw outer glow
        outer_gradient = QRadialGradient(center_x, center_y, glow_radius)
        
        outer_color = QColor(self.color)
        outer_color.setAlpha(int(100 * pulse))
        outer_gradient.setColorAt(0, outer_color)
        
        outer_fade = QColor(self.color)
        outer_fade.setAlpha(0)
        outer_gradient.setColorAt(1, outer_fade)
        
        painter.setBrush(QBrush(outer_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(center_x, center_y), glow_radius, glow_radius)
        
        # Draw core orb
        core_gradient = QRadialGradient(center_x, center_y, radius)
        
        core_bright = QColor(self.color).lighter(150)
        core_gradient.setColorAt(0, core_bright)
        core_gradient.setColorAt(0.7, self.color)
        
        core_edge = QColor(self.color)
        core_edge.setAlpha(200)
        core_gradient.setColorAt(1, core_edge)
        
        painter.setBrush(QBrush(core_gradient))
        painter.drawEllipse(QPointF(center_x, center_y), radius, radius)
        
        painter.end()


class BarWaveLoader(QWidget):
    """
    Wave effect with bars - dynamic and rhythmic.
    Modern and professional.
    """
    
    def __init__(self, parent=None, size=100, num_bars=7, color=None):
        super().__init__(parent)
        
        self.loader_size = size
        self.num_bars = num_bars
        self.color = color or QColor(34, 197, 94)  # Green
        self.phase = 0
        
        self.setFixedSize(size, size)
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.setInterval(16)
    
    def start(self):
        self.timer.start()
        self.show()
    
    def stop(self):
        self.timer.stop()
        self.hide()
    
    def animate(self):
        self.phase = (self.phase + 4) % 360
        self.update()
    
    def paintEvent(self, event):
        """Draw wave bars"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        bar_width = 8
        max_height = self.loader_size / 2
        min_height = self.loader_size / 8
        spacing = 12
        
        total_width = (self.num_bars * bar_width) + ((self.num_bars - 1) * spacing)
        start_x = (self.width() - total_width) / 2
        base_y = self.height() / 2
        
        for i in range(self.num_bars):
            x = start_x + i * (bar_width + spacing)
            
            # Calculate height based on sine wave
            wave_offset = (i * 30)  # Offset between bars
            wave = math.sin((self.phase + wave_offset) * math.pi / 180)
            height = min_height + ((max_height - min_height) * ((wave + 1) / 2))
            
            # Color intensity based on height
            intensity = (wave + 1) / 2
            bar_color = QColor(self.color)
            bar_color.setAlpha(int(150 + (105 * intensity)))
            
            # Draw bar
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(bar_color))
            painter.drawRoundedRect(int(x), int(base_y - height/2), int(bar_width), int(height), 4, 4)
            
            # Draw glow
            glow_color = QColor(self.color)
            glow_color.setAlpha(int(50 * intensity))
            painter.setBrush(QBrush(glow_color))
            painter.drawRoundedRect(int(x - 2), int(base_y - height/2 - 2), 
                                   int(bar_width + 4), int(height + 4), 5, 5)
        
        painter.end()


# Test application
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QLabel
    import sys
    
    app = QApplication(sys.argv)
    
    window = QWidget()
    window.setWindowTitle("TruScore Premium Loaders Test")
    window.resize(900, 600)
    window.setStyleSheet("background-color: #0f172a;")
    
    layout = QVBoxLayout(window)
    layout.setSpacing(40)
    layout.setContentsMargins(50, 50, 50, 50)
    
    # Row 1
    row1 = QHBoxLayout()
    
    orbiting = OrbitingBallsLoader(size=100, color=QColor(56, 189, 248))
    orbiting.start()
    row1.addWidget(QLabel("<font color='white'>Orbiting Balls</font>"))
    row1.addWidget(orbiting)
    row1.addStretch()
    
    domino = DominoLoader(size=100, color=QColor(255, 255, 255))
    domino.start()
    row1.addWidget(QLabel("<font color='white'>Domino Fall</font>"))
    row1.addWidget(domino)
    
    layout.addLayout(row1)
    
    # Row 2
    row2 = QHBoxLayout()
    
    pulse_ring = PulseRingLoader(size=100, color=QColor(168, 85, 247))
    pulse_ring.start()
    row2.addWidget(QLabel("<font color='white'>Pulse Rings</font>"))
    row2.addWidget(pulse_ring)
    row2.addStretch()
    
    glow = GlowPulseLoader(size=100, color=QColor(236, 72, 153))
    glow.start()
    row2.addWidget(QLabel("<font color='white'>Glow Pulse</font>"))
    row2.addWidget(glow)
    
    layout.addLayout(row2)
    
    # Row 3
    row3 = QHBoxLayout()
    
    wave = BarWaveLoader(size=100, color=QColor(34, 197, 94))
    wave.start()
    row3.addWidget(QLabel("<font color='white'>Bar Wave</font>"))
    row3.addWidget(wave)
    row3.addStretch()
    
    layout.addLayout(row3)
    
    layout.addStretch()
    
    window.show()
    sys.exit(app.exec())
