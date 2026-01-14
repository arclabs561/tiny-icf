"""Aim experiment tracking integration for ICF model training."""

from pathlib import Path
from typing import Any, Optional

try:
    from aim import Run
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    Run = None


class AimTracker:
    """Wrapper for Aim Run that handles optional dependency and provides convenient methods."""
    
    def __init__(
        self,
        experiment_name: str = "icf-training",
        run_name: Optional[str] = None,
        repo: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize Aim tracker.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Optional name for this specific run
            repo: Path to Aim repository (defaults to ~/.aim)
            enabled: Whether tracking is enabled (useful for disabling in tests)
        """
        self.enabled = enabled and AIM_AVAILABLE
        self.run: Optional[Run] = None
        
        if self.enabled:
            try:
                # If run_name is provided, try to use it; otherwise let Aim generate one
                if run_name:
                    self.run = Run(experiment=experiment_name, run_hash=run_name, repo=repo)
                else:
                    self.run = Run(experiment=experiment_name, repo=repo)
            except Exception as e:
                print(f"Warning: Failed to initialize Aim tracker: {e}")
                # Try without run_hash if that was the issue
                try:
                    self.run = Run(experiment=experiment_name, repo=repo)
                except Exception as e2:
                    print(f"Warning: Failed to initialize Aim tracker (fallback): {e2}")
                    self.enabled = False
                    self.run = None
    
    def track_params(self, params: dict[str, Any]) -> None:
        """Track hyperparameters."""
        if not self.enabled or not self.run:
            return
        
        try:
            self.run["hparams"] = params
        except Exception as e:
            print(f"Warning: Failed to track params: {e}")
    
    def track_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Track a metric value."""
        if not self.enabled or not self.run:
            return
        
        try:
            self.run.track(value, name=name, step=step, context=context or {})
        except Exception as e:
            print(f"Warning: Failed to track metric {name}: {e}")
    
    def track_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Track multiple metrics at once."""
        if not self.enabled or not self.run:
            return
        
        for name, value in metrics.items():
            self.track_metric(name, value, step=step)
    
    def track_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        """Track text data."""
        if not self.enabled or not self.run:
            return
        
        try:
            self.run.track(text, name=name, step=step, context={"type": "text"})
        except Exception as e:
            print(f"Warning: Failed to track text {name}: {e}")
    
    def track_dict(self, name: str, data: dict[str, Any], step: Optional[int] = None) -> None:
        """Track a dictionary as a single object."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Aim can track dictionaries directly
            self.run[name] = data
        except Exception as e:
            print(f"Warning: Failed to track dict {name}: {e}")
    
    def close(self) -> None:
        """Close the run and finalize tracking."""
        if self.enabled and self.run:
            try:
                self.run.close()
            except Exception as e:
                print(f"Warning: Failed to close Aim run: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def set_tags(self, tags: list[str]) -> None:
        """Set tags for the run."""
        if not self.enabled or not self.run:
            return
        
        try:
            for tag in tags:
                self.run.add_tag(tag)
        except Exception as e:
            print(f"Warning: Failed to set tags: {e}")
    
    def set_properties(self, properties: dict[str, Any]) -> None:
        """Set properties (metadata) for the run."""
        if not self.enabled or not self.run:
            return
        
        try:
            for key, value in properties.items():
                self.run[key] = value
        except Exception as e:
            print(f"Warning: Failed to set properties: {e}")
    
    def track_breadcrumb(self, breadcrumb: str, step: Optional[int] = None) -> None:
        """Track a breadcrumb (note/annotation) for the run."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Track as text with breadcrumb context
            self.run.track(breadcrumb, name='breadcrumb', step=step, context={'type': 'breadcrumb'})
        except Exception as e:
            print(f"Warning: Failed to track breadcrumb: {e}")
    
    def track_note(self, note: str, step: Optional[int] = None) -> None:
        """Track a note (detailed annotation) for the run."""
        if not self.enabled or not self.run:
            return
        
        try:
            # Track as text with note context
            self.run.track(note, name='note', step=step, context={'type': 'note'})
        except Exception as e:
            print(f"Warning: Failed to track note: {e}")
    
    def track_breadcrumbs(self, breadcrumbs: list[str], step: Optional[int] = None) -> None:
        """Track multiple breadcrumbs at once."""
        for breadcrumb in breadcrumbs:
            self.track_breadcrumb(breadcrumb, step=step)
    
    def __bool__(self) -> bool:
        """Check if tracker is enabled and available."""
        return self.enabled and self.run is not None

