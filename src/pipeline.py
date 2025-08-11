class WrapperPipeline:
    def __init__(self):
        self.available_operations = {}
        self.active_pipeline_steps = []
        self.operation_enabled_status = {}

    def register_operation(self, operation, name, default_enabled=True):
        if name in self.available_operations:
            raise Exception(f"Operation '{name}' is already registered!")
        self.available_operations[name] = operation
        self.operation_enabled_status[name] = default_enabled
        self.rebuild_pipeline()

    def remove_operation(self, name):
        if name in self.available_operations:
            del self.available_operations[name]

    def enable_operation(self, name):
        if name not in self.available_operations:
            raise Exception(f"Can not enable operation '{name}' because it is not registered!")
        self.operation_enabled_status[name] = True
        self.rebuild_pipeline()

    def disable_operation(self, name):
        if name not in self.available_operations:
            raise Exception(f"Can not disable operation '{name}' because it is not registered!")
        self.operation_enabled_status[name] = False
        self.rebuild_pipeline()

    def rebuild_pipeline(self):
        self.active_pipeline_steps = []
        for name, operation in self.available_operations.items():
            if self.operation_enabled_status[name]:
                self.active_pipeline_steps.append(operation)

    def process(self, *args):
        current_data = args
        for operation in self.active_pipeline_steps:
            current_data = operation(current_data)
        return current_data
