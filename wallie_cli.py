#!/usr/bin/env python3
"""
Wallie Voice Bot CLI - Monitor and control Wallie
"""

import typer
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import psutil
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich import box

app = typer.Typer(help="Wallie Voice Bot CLI")
console = Console()

def get_log_path() -> Path:
    """Get the log file path"""
    return Path.home() / ".wallie_voice_bot" / "logs" / "wallie.jsonl"

def get_pid_file() -> Path:
    """Get the PID file path"""
    return Path.home() / ".wallie_voice_bot" / "wallie.pid"

def is_running() -> bool:
    """Check if Wallie is running"""
    pid_file = get_pid_file()
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            return psutil.pid_exists(pid)
        except:
            pass
    return False

@app.command()
def status():
    """Show Wallie's current status"""
    if is_running():
        console.print("[green]✓ Wallie is running[/green]")
        
        # Get process info
        pid_file = get_pid_file()
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            try:
                process = psutil.Process(pid)
                console.print(f"PID: {pid}")
                console.print(f"CPU: {process.cpu_percent()}%")
                console.print(f"Memory: {process.memory_info().rss / 1024**2:.1f} MB")
                console.print(f"Uptime: {datetime.now() - datetime.fromtimestamp(process.create_time())}")
            except:
                pass
    else:
        console.print("[red]✗ Wallie is not running[/red]")

@app.command()
def start():
    """Start Wallie daemon"""
    if is_running():
        console.print("[yellow]Wallie is already running[/yellow]")
        return
    
    console.print("Starting Wallie...")
    subprocess.run([sys.executable, "wallie_voice_bot.py", "--daemon"])
    
    # Wait for startup
    for _ in range(10):
        if is_running():
            console.print("[green]✓ Wallie started successfully[/green]")
            return
        time.sleep(1)
    
    console.print("[red]Failed to start Wallie[/red]")

@app.command()
def stop():
    """Stop Wallie daemon"""
    if not is_running():
        console.print("[yellow]Wallie is not running[/yellow]")
        return
    
    pid_file = get_pid_file()
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        console.print(f"Stopping Wallie (PID: {pid})...")
        
        try:
            process = psutil.Process(pid)
            process.terminate()
            process.wait(timeout=5)
            console.print("[green]✓ Wallie stopped[/green]")
        except psutil.TimeoutExpired:
            console.print("[yellow]Force killing Wallie...[/yellow]")
            process.kill()
        except Exception as e:
            console.print(f"[red]Error stopping Wallie: {e}[/red]")

@app.command()
def restart():
    """Restart Wallie daemon"""
    stop()
    time.sleep(2)
    start()

@app.command()
def logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(20, "--lines", "-n", help="Number of lines to show"),
    stage: Optional[str] = typer.Option(None, "--stage", "-s", help="Filter by stage (vad, asr, llm, tts)")
):
    """View Wallie logs"""
    log_path = get_log_path()
    
    if not log_path.exists():
        console.print("[red]No log file found[/red]")
        return
    
    if follow:
        # Follow mode
        console.print(f"Following {log_path} (Ctrl+C to stop)...")
        try:
            with open(log_path, 'r') as f:
                # Go to end
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        try:
                            log_entry = json.loads(line)
                            if stage and log_entry.get('stage') != stage:
                                continue
                            
                            # Format output
                            timestamp = log_entry.get('timestamp', '')
                            level = log_entry.get('level', 'INFO')
                            stage_name = log_entry.get('stage', 'main')
                            message = log_entry.get('message', '')
                            
                            color = {
                                'ERROR': 'red',
                                'WARNING': 'yellow',
                                'INFO': 'blue'
                            }.get(level, 'white')
                            
                            console.print(
                                f"[dim]{timestamp}[/dim] "
                                f"[{color}]{level:7}[/{color}] "
                                f"[cyan]{stage_name:6}[/cyan] "
                                f"{message}"
                            )
                            
                            # Show metrics if present
                            if log_entry.get('latency_ms'):
                                console.print(f"  └─ Latency: {log_entry['latency_ms']:.1f}ms", style="dim")
                        except:
                            console.print(line.strip())
                    else:
                        time.sleep(0.1)
        except KeyboardInterrupt:
            console.print("\nStopped following logs")
    else:
        # Show last N lines
        with open(log_path, 'r') as f:
            all_lines = f.readlines()
            
        # Filter and show
        shown = 0
        for line in reversed(all_lines):
            if shown >= lines:
                break
                
            try:
                log_entry = json.loads(line)
                if stage and log_entry.get('stage') != stage:
                    continue
                
                # Format output
                timestamp = log_entry.get('timestamp', '')
                level = log_entry.get('level', 'INFO')
                stage_name = log_entry.get('stage', 'main')
                message = log_entry.get('message', '')
                
                color = {
                    'ERROR': 'red',
                    'WARNING': 'yellow',
                    'INFO': 'blue'
                }.get(level, 'white')
                
                console.print(
                    f"[dim]{timestamp}[/dim] "
                    f"[{color}]{level:7}[/{color}] "
                    f"[cyan]{stage_name:6}[/cyan] "
                    f"{message}"
                )
                shown += 1
            except:
                pass

@app.command()
def metrics():
    """Show performance metrics"""
    log_path = get_log_path()
    
    if not log_path.exists():
        console.print("[red]No log file found[/red]")
        return
    
    # Collect metrics from last hour
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    
    metrics_data = {
        'vad': {'count': 0, 'wake_words': 0},
        'asr': {'count': 0, 'latencies': []},
        'llm': {'count': 0, 'latencies': [], 'tokens': 0},
        'tts': {'count': 0, 'latencies': []}
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                timestamp = datetime.fromisoformat(entry.get('timestamp', ''))
                
                if timestamp < one_hour_ago:
                    continue
                
                stage = entry.get('stage')
                if stage in metrics_data:
                    metrics_data[stage]['count'] += 1
                    
                    if stage == 'vad' and 'wake_word_count' in entry:
                        metrics_data['vad']['wake_words'] = entry['wake_word_count']
                    
                    if 'latency_ms' in entry and entry['latency_ms']:
                        if stage in ['asr', 'llm', 'tts']:
                            metrics_data[stage]['latencies'].append(entry['latency_ms'])
                    
                    if stage == 'llm' and 'total_tokens' in entry:
                        metrics_data['llm']['tokens'] = entry['total_tokens']
            except:
                pass
    
    # Create metrics table
    table = Table(title="Performance Metrics (Last Hour)", box=box.ROUNDED)
    table.add_column("Stage", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Avg Latency (ms)", justify="right")
    table.add_column("P95 Latency (ms)", justify="right")
    table.add_column("Notes", style="dim")
    
    # VAD metrics
    table.add_row(
        "VAD",
        str(metrics_data['vad']['count']),
        "-",
        "-",
        f"Wake words: {metrics_data['vad']['wake_words']}"
    )
    
    # Other stages
    for stage in ['asr', 'llm', 'tts']:
        latencies = metrics_data[stage]['latencies']
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            
            notes = ""
            if stage == 'llm':
                notes = f"Tokens: {metrics_data['llm']['tokens']}"
            
            table.add_row(
                stage.upper(),
                str(metrics_data[stage]['count']),
                f"{avg_latency:.1f}",
                f"{p95_latency:.1f}",
                notes
            )
        else:
            table.add_row(
                stage.upper(),
                str(metrics_data[stage]['count']),
                "-",
                "-",
                ""
            )
    
    console.print(table)

@app.command()
def monitor():
    """Live monitoring dashboard"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    
    def generate_dashboard():
        # Header
        if is_running():
            status_text = "[green]● RUNNING[/green]"
        else:
            status_text = "[red]● STOPPED[/red]"
        
        header = Panel(
            f"Wallie Voice Bot Monitor {status_text}",
            style="bold white on blue"
        )
        layout["header"].update(header)
        
        # Main content
        if is_running():
            # Get recent logs
            log_path = get_log_path()
            recent_logs = []
            
            if log_path.exists():
                with open(log_path, 'r') as f:
                    lines = f.readlines()[-10:]  # Last 10 lines
                    for line in lines:
                        try:
                            entry = json.loads(line)
                            recent_logs.append(
                                f"[{entry.get('stage', 'main'):6}] {entry.get('message', '')}"
                            )
                        except:
                            pass
            
            main_content = "\n".join(recent_logs) if recent_logs else "No recent activity"
        else:
            main_content = "Wallie is not running\n\nStart with: wallie-cli start"
        
        layout["main"].update(Panel(main_content, title="Recent Activity"))
        
        # Footer
        footer = Panel(
            "Press Ctrl+C to exit | R to restart | S to stop",
            style="dim"
        )
        layout["footer"].update(footer)
        
        return layout
    
    try:
        with Live(generate_dashboard(), refresh_per_second=2) as live:
            while True:
                time.sleep(0.5)
                live.update(generate_dashboard())
    except KeyboardInterrupt:
        console.print("\nMonitoring stopped")

if __name__ == "__main__":
    import sys
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    app()
