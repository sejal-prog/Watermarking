# Copyright (c) Sejal Jadhav. Licensed under the MIT License.
# This module extends Meta's VideoSeal (https://github.com/facebookresearch/videoseal)
# with group-equivariant CNN support for rotation-invariant watermarking.

import logging

logger = logging.getLogger(__name__)

class RotationScheduler:
    """
    Progressive rotation augmentation scheduler for curriculum learning.
    
    Gradually increases the rotation angle range from start_angle to end_angle
    over the course of training epochs.
    """
    
    def __init__(
        self,
        start_angle: float = 10,
        end_angle: float = 45,
        start_epoch: int = 0,
        end_epoch: int = 600,
        schedule_type: str = 'linear'
    ):
        """
        Args:
            start_angle: Initial max rotation angle (degrees)
            end_angle: Final max rotation angle (degrees)
            start_epoch: Epoch to start progressive schedule
            end_epoch: Epoch to reach end_angle
            schedule_type: Type of schedule ('linear', 'cosine', 'step', 'exponential')
        """
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.schedule_type = schedule_type
        
        assert end_angle >= start_angle, "end_angle must be >= start_angle"
        assert end_epoch > start_epoch, "end_epoch must be > start_epoch"
        
        logger.info(
            "RotationScheduler initialized: type=%s, angles=%s°→%s°, epochs=%s→%s",
            schedule_type, start_angle, end_angle, start_epoch, end_epoch,
        )
    
    def get_rotation_range(self, epoch: int) -> float:
        """
        Get the current rotation range for this epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Current maximum rotation angle (degrees)
        """
        if epoch < self.start_epoch:
            return self.start_angle
        elif epoch >= self.end_epoch:
            return self.end_angle
        
        # Calculate progress [0, 1]
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        
        # Apply schedule
        if self.schedule_type == 'linear':
            factor = progress
        elif self.schedule_type == 'cosine':
            factor = self._cosine_schedule(progress)
        elif self.schedule_type == 'step':
            factor = self._step_schedule(progress)
        elif self.schedule_type == 'exponential':
            factor = self._exponential_schedule(progress)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        current_angle = self.start_angle + factor * (self.end_angle - self.start_angle)
        return current_angle
    
    def _cosine_schedule(self, progress: float) -> float:
        """
        Cosine annealing schedule: slow start, fast middle, slow end
        """
        return (1 - math.cos(progress * math.pi)) / 2
    
    def _step_schedule(self, progress: float) -> float:
        """
        Step schedule: discrete jumps every 25%
        """
        if progress < 0.25:
            return 0.0
        elif progress < 0.5:
            return 0.33
        elif progress < 0.75:
            return 0.66
        else:
            return 1.0
    
    def _exponential_schedule(self, progress: float) -> float:
        """
        Exponential schedule: fast start, slow end
        """
        return (math.exp(progress * 2) - 1) / (math.exp(2) - 1)
    
    def get_schedule_info(self, total_epochs: int) -> str:
        """
        Get schedule info for logging.
        """
        milestones = [0, total_epochs // 4, total_epochs // 2, 
                     3 * total_epochs // 4, total_epochs - 1]
        info = "Rotation Schedule:\n"
        for epoch in milestones:
            angle = self.get_rotation_range(epoch)
            info += f"  Epoch {epoch:3d}: ±{angle:.1f}°\n"
        return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scheduler = RotationScheduler(
        start_angle=10, end_angle=45,
        start_epoch=0, end_epoch=600,
        schedule_type="linear",
    )
    logger.info("\n%s", scheduler.get_schedule_info(600))

    for epoch in [0, 150, 300, 450, 600]:
        angle = scheduler.get_rotation_range(epoch)
        logger.info("Epoch %d: ±%.2f°", epoch, angle)