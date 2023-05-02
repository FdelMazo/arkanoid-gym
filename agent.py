from abc import ABC, abstractmethod


class ArkAgent(ABC):
    @abstractmethod
    def get_action(self, screen, info):
        pass

    @abstractmethod
    def update(self, screen, info, action, reward, done, next_screen, next_info):
        pass
