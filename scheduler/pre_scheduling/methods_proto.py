"""
算法接口定义
定义了工作流排序和截止时间划分的基础接口
"""


class WorkflowSequencing:
    """工作流排序算法基类"""
    
    def run(self, workflows, system):
        """
        运行工作流排序算法
        
        Args:
            workflows: 工作流列表
            system: 系统资源配置（如vms等）
            
        Returns:
            排序后的工作流列表
        """
        raise NotImplementedError


class DeadlinePartition:
    """截止时间划分算法基类"""
    
    def run(self, workflow, system):
        """
        运行截止时间划分算法
        
        Args:
            workflow: 单个工作流
            system: 系统资源配置（如vms等）
        """
        raise NotImplementedError

