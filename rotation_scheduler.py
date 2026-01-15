import random


class ProgressiveRotationScheduler:
    """
    Manages progressive curriculum for rotation training with experience replay.
    
    Training Schedule:
    - Phase 1 (epochs 0-199):   Train on 10-15° only
    - Phase 2 (epochs 200-399): 70% on 20-30° + 30% replay 10-15°
    - Phase 3 (epochs 400-600): 50% on 35-45° + 30% replay 10-15° + 20% replay 20-30°
    
    Args:
        max_epochs: Total number of training epochs (default: 600)
        use_replay: Enable experience replay to prevent catastrophic forgetting (default: True)
    """
    
    def __init__(self, max_epochs=600, use_replay=True):
        self.max_epochs = max_epochs
        self.current_epoch = 0
        self.use_replay = use_replay
        
        
        self.phase2_new_prob = 0.7   
        self.phase2_replay_prob = 0.3  
        
        self.phase3_new_prob = 0.5      
        self.phase3_basics_prob = 0.3   
        self.phase3_medium_prob = 0.2   
    
    def get_rotation_angle(self, epoch):
        """
        Returns rotation angle for given epoch.
        
        With replay enabled: Probabilistically samples from multiple difficulty ranges
        to prevent catastrophic forgetting of easy rotations.
        
        Args:
            epoch: Current training epoch (0-600)
            
        Returns:
            angle: Rotation angle in degrees (float)
        """
        
        # Without replay: Original progressive behavior
        if not self.use_replay:
            if epoch < 200:
                return random.uniform(10.0, 15.0)
            elif epoch < 400:
                return random.uniform(20.0, 30.0)
            else:
                return random.uniform(35.0, 45.0)
            
        #Now with replay
        # 100% training on easy rotations
        if epoch < 200:
            return random.uniform(10.0, 15.0)
        
        # Phase 2: Learn medium + REPLAY basics (epochs 200-399)
        # 70% new material, 30% replay to maintain basic skills
        elif epoch < 400:
            rand = random.random()
            
            if rand < self.phase2_new_prob:
                # Learn NEW material (medium difficulty)
                return random.uniform(20.0, 30.0)
            else:
                # REPLAY old material (easy difficulty)
                return random.uniform(10.0, 15.0)
        
        # Phase 3: Learn hard + REPLAY all previous (epochs 400-600)
        # 50% new material, 30% replay basics, 20% replay medium
        else:
            rand = random.random()
            
            if rand < self.phase3_new_prob:
                # Learn NEW material (hard difficulty)
                return random.uniform(35.0, 45.0)
            elif rand < (self.phase3_new_prob + self.phase3_basics_prob):
                # REPLAY basics (easy difficulty)
                return random.uniform(10.0, 15.0)
            else:
                # REPLAY medium difficulty
                return random.uniform(20.0, 30.0)
    
    def update_epoch(self, epoch):
        """
        Update current epoch and log training phase.
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        phase_str = self.get_phase_string(epoch)
        print(f"Epoch {epoch}/{self.max_epochs}: {phase_str}")
    
    def get_phase_string(self, epoch):
        """
        Get human-readable string describing current training phase.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            phase_str: Description of current phase
        """
        if not self.use_replay:
            # Original progressive learning (no replay)
            if epoch < 200:
                return "Phase 1: 10-15° (easy)"
            elif epoch < 400:
                return "Phase 2: 20-30° (medium)"
            else:
                return "Phase 3: 35-45° (hard)"
        else:
            # With experience replay
            if epoch < 200:
                return "Phase 1: 10-15° (easy) [100%]"
            elif epoch < 400:
                return f"Phase 2: 20-30° (new) [{self.phase2_new_prob*100:.0f}%] + 10-15° (replay) [{self.phase2_replay_prob*100:.0f}%]"
            else:
                return f"Phase 3: 35-45° (new) [{self.phase3_new_prob*100:.0f}%] + 10-15° (replay) [{self.phase3_basics_prob*100:.0f}%] + 20-30° (replay) [{self.phase3_medium_prob*100:.0f}%]"
    
    def get_statistics(self, num_samples=1000):
        """
        Get statistics about angle distribution for current epoch.
        Useful for debugging and validation.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            stats: Dictionary with angle distribution statistics
        """
        angles = [self.get_rotation_angle(self.current_epoch) for _ in range(num_samples)]
        
        # Count angles in different ranges
        easy_count = sum(1 for a in angles if 10 <= a <= 15)
        medium_count = sum(1 for a in angles if 20 <= a <= 30)
        hard_count = sum(1 for a in angles if 35 <= a <= 45)
        
        return {
            'easy_pct': easy_count / num_samples * 100,
            'medium_pct': medium_count / num_samples * 100,
            'hard_pct': hard_count / num_samples * 100,
            'mean_angle': sum(angles) / num_samples,
            'min_angle': min(angles),
            'max_angle': max(angles)
        }

if __name__ == "__main__":
    print("="*80)
    print("TESTING EXPERIENCE REPLAY IMPLEMENTATION")
    print("="*80)
    
    # Create scheduler with replay
    scheduler = ProgressiveRotationScheduler(max_epochs=600, use_replay=True)
    
    # Test different epochs
    test_epochs = [50, 150, 250, 350, 450, 550]
    
    for epoch in test_epochs:
        print(f"\n{'='*80}")
        scheduler.update_epoch(epoch)
        
        # Get statistics for this epoch
        stats = scheduler.get_statistics(num_samples=1000)
        
        print(f"\nAngle distribution (1000 samples):")
        print(f"  Easy (10-15°):   {stats['easy_pct']:5.1f}%")
        print(f"  Medium (20-30°): {stats['medium_pct']:5.1f}%")
        print(f"  Hard (35-45°):   {stats['hard_pct']:5.1f}%")
        print(f"\nAngle range: {stats['min_angle']:.1f}° - {stats['max_angle']:.1f}°")
        print(f"Mean angle: {stats['mean_angle']:.1f}°")
        
        # Sample a few angles to show
        print(f"\nSample angles:")
        for i in range(5):
            angle = scheduler.get_rotation_angle(epoch)
            angle_type = "EASY" if 10 <= angle <= 15 else "MEDIUM" if 20 <= angle <= 30 else "HARD"
            print(f"  {angle:.2f}° ({angle_type})")
    
    print("\n" + "="*80)
    print("COMPARISON: Without vs With Replay")
    print("="*80)
    
    # Compare with and without replay
    scheduler_no_replay = ProgressiveRotationScheduler(max_epochs=600, use_replay=False)
    scheduler_with_replay = ProgressiveRotationScheduler(max_epochs=600, use_replay=True)
    
    for epoch in [250, 450]:
        print(f"\n{'─'*80}")
        print(f"Epoch {epoch}:")
        print(f"{'─'*80}")
        
        # Without replay
        scheduler_no_replay.update_epoch(epoch)
        stats_no = scheduler_no_replay.get_statistics(1000)
        print(f"\nWITHOUT replay:")
        print(f"  Easy: {stats_no['easy_pct']:.1f}%  Medium: {stats_no['medium_pct']:.1f}%  Hard: {stats_no['hard_pct']:.1f}%")
        
        # With replay
        scheduler_with_replay.update_epoch(epoch)
        stats_yes = scheduler_with_replay.get_statistics(1000)
        print(f"\nWITH replay:")
        print(f"  Easy: {stats_yes['easy_pct']:.1f}%  Medium: {stats_yes['medium_pct']:.1f}%  Hard: {stats_yes['hard_pct']:.1f}%")
        
        print(f"\nDifference:")
        print(f"  Easy: {stats_yes['easy_pct'] - stats_no['easy_pct']:+.1f}%  ← Replay keeps basics fresh!")