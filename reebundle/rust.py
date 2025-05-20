"""
Rust implementation of reebundle functions.
This file provides access to the compiled Rust code.
"""

# These will be provided by the compiled Rust extension
from reebundle.rust import Event, find_connect_disconnect_events

__all__ = ["Event", "find_connect_disconnect_events"] 