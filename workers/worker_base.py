class WorkerBase:
    """Base class for all workers with async support"""
    
    def __init__(self, name: str, config: Dict[str, Any], queues: Dict[str, mp.Queue]):
        self.name = name
        self.config = config
        self.queues = queues
        self.control_queue = queues[f'{name}_control']
        self.logger = self._setup_logging()
        self.running = True
    
    async def handle_control_messages(self):
        """Base control message handler"""
        while self.running:
            try:
                msg = await self.get_control_message()
                if msg:
                    await self.process_control_message(msg)
            except Exception as e:
                self.logger.error(f"Control message error: {e}")
            await asyncio.sleep(0.01)
    
    async def get_control_message(self) -> Optional[Dict[str, Any]]:
        """Non-blocking control message check"""
        try:
            return self.control_queue.get_nowait()
        except Queue.Empty:
            return None
    
    async def process_control_message(self, msg: Dict[str, Any]):
        """Process control messages"""
        msg_type = msg.get('type')
        
        if msg_type == 'abort':
            await self.handle_abort()
        elif msg_type == 'config_reload':
            await self.handle_config_reload(msg.get('config', {}))
    
    async def handle_abort(self):
        """Override in subclasses"""
        raise NotImplementedError
    
    async def handle_config_reload(self, new_config: Dict[str, Any]):
        """Override in subclasses"""
        raise NotImplementedError
    
    def signal_ready(self):
        """Signal worker is ready"""
        self.control_queue.put({
            'type': 'ready',
            'worker': self.name,
            'timestamp': time.time()
        })