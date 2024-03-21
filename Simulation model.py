import simpy
import random
import matplotlib.pyplot as plt
from matplotlib import ticker
# 服务时间定义函数
def fill_time():
    return random.uniform(200, 350)

def level_time():
    return random.uniform(900, 950)

def watering_time():
    return random.uniform(500, 600)

def compact_time():
    return random.uniform(1000, 1500)

# 创建存储实体的字典
entities = {}

# 统计类
class Statistics:
    def __init__(self):
        self.total_wait_time = 0
        self.total_entities = 0
        self.total_fill_queue_length = 0
        self.total_level_queue_length = 0
        self.total_watering_queue_length = 0
        self.total_compact_queue_length = 0
        self.total_service_time = 0

    def update_wait_time(self, wait_time):
        self.total_wait_time += wait_time

    def update_service_time(self, service_time):
        self.total_service_time += service_time

    def increment_entities(self):
        self.total_entities += 1

    def update_fill_queue_length(self, length):
        self.total_fill_queue_length += length

    def update_level_queue_length(self, length):
        self.total_level_queue_length += length

    def update_watering_queue_length(self, length):
        self.total_watering_queue_length += length

    def update_compact_queue_length(self, length):
        self.total_compact_queue_length += length

    def average_queue_length(self, num_queues):
        total_queue_length = (self.total_fill_queue_length + self.total_level_queue_length +
                              self.total_watering_queue_length + self.total_compact_queue_length)
        return total_queue_length / num_queues

    def average_wait_time(self):
        return self.total_wait_time / self.total_entities if self.total_entities != 0 else 0

# 土层实体类
class SoilEntity:
    def __init__(self, name,cengshu):
        self.name = name
        self.service_times = {'fill': fill_time(),
                              'level': level_time(),
                              'watering': watering_time(),
                              'compact': compact_time()}
        self.intermission_times = {'fill': random.uniform(10, 30),
                                   'level': random.uniform(10, 30),
                                   'watering': random.uniform(10, 30),
                                   'compact': random.uniform(10, 30)}
        self.arrival_times = {'fill': 0, 'level': 0, 'watering': 0, 'compact': 0}
        self.chengsu = cengshu
        self.stage = None
        self.served_times = {'fill': 0, 'level': 0, 'watering': 0, 'compact': 0}
        self.arrival_time = 0
        self.is_being_serviced = False

# 服务台状态类
class ServiceStage:
    def __init__(self, env):
        self.resource = simpy.Resource(env, capacity=1)

# 各阶段服务台
def fill_stage(env, soil_entity, level_stage_queue, service_stage, stats):
    with service_stage.resource.request() as request:
        yield request
        soil_entity.process = env.active_process
        wait_time = env.now - soil_entity.arrival_time
        start_time = env.now
        print(f"{env.now}: {soil_entity.name,soil_entity.chengsu} 开始填筑阶段")
        yield env.timeout(soil_entity.service_times['fill'])
        service_time = env.now - start_time
        soil_entity.served_times['fill'] = service_time
        stats.update_service_time(service_time)
        print(f"{env.now}: {soil_entity.name,soil_entity.chengsu} 完成填筑阶段，服务时间：{service_time}")
        env.process(intermission_stage(env, soil_entity, level_stage_queue, 'fill'))

def level_stage(env, soil_entity, compact_stage_queue, service_stage, stats):
    with service_stage.resource.request() as request:
        yield request
        soil_entity.process = env.active_process
        wait_time = env.now - soil_entity.arrival_time
        stats.update_wait_time(wait_time)
        start_time = env.now
        print(f"{env.now}: {soil_entity.name,soil_entity.chengsu} 开始平整阶段")
        yield env.timeout(soil_entity.service_times['level'])
        service_time = env.now - start_time
        soil_entity.served_times['level'] = service_time
        stats.update_service_time(service_time)
        print(f"{env.now}: {soil_entity.name,soil_entity.chengsu} 完成平整阶段，服务时间：{service_time}")
        env.process(intermission_stage(env, soil_entity, compact_stage_queue, 'level'))

def watering_stage(env, soil_entity, compact_stage_queue, service_stage, stats):
    with service_stage.resource.request() as request:
        yield request
        soil_entity.process = env.active_process
        wait_time = env.now - soil_entity.arrival_time
        stats.update_wait_time(wait_time)
        start_time = env.now
        print(f"{env.now}: {soil_entity.name,soil_entity.chengsu} 开始洒水晾晒阶段")
        yield env.timeout(soil_entity.service_times['watering'])
        service_time = env.now - start_time
        soil_entity.served_times['watering'] = service_time
        stats.update_service_time(service_time)
        print(f"{env.now}: {soil_entity.name,soil_entity.chengsu} 完成洒水晾晒阶段，服务时间：{service_time}")
        env.process(intermission_stage(env, soil_entity, compact_stage_queue, 'watering'))

def compact_stage(env, soil_entity,fill_stage_queue, service_stage, stats):
    with service_stage.resource.request() as request:
        yield request
        soil_entity.process = env.active_process
        wait_time = env.now - soil_entity.arrival_time
        stats.update_wait_time(wait_time)
        start_time = env.now
        print(f"{env.now}: {soil_entity.name,soil_entity.chengsu} 开始压实阶段")
        yield env.timeout(soil_entity.service_times['compact'])
        service_time = env.now - start_time
        soil_entity.served_times['compact'] = service_time
        stats.update_service_time(service_time)
        print(f"{env.now}: {soil_entity.name,soil_entity.chengsu} 完成压实阶段，服务时间：{service_time}")
        if soil_entity.chengsu < 12:
            env.process(intermission_stage(env, soil_entity, fill_stage_queue, 'compact'))
            a = soil_entity.chengsu + 1
            soil_entity.chengsu = a


# 间歇阶段
def intermission_stage(env, soil_entity, next_stage_queue, stage_name):
    yield env.timeout(soil_entity.intermission_times[stage_name])
    next_stage_queue.put(soil_entity)

# 土层处理过程
def fill_stage_process(env, fill_stage_queue, level_stage_queue, service_stages, entities, stats):
    while True:
        soil_entity = yield fill_stage_queue.get()
        soil_entity.arrival_time = env.now
        soil_entity.stage = 'fill'
        stats.increment_entities()
        env.process(fill_stage(env, soil_entity, level_stage_queue, service_stages['fill'], stats))

def level_stage_process(env, level_stage_queue, watering_stage_queue, service_stages, entities, stats):
    while True:
        soil_entity = yield level_stage_queue.get()
        soil_entity.arrival_time = env.now
        soil_entity.stage = 'level'
        stats.increment_entities()
        env.process(level_stage(env, soil_entity, watering_stage_queue, service_stages['level'], stats))

def watering_stage_process(env, watering_stage_queue, compact_stage_queue, service_stages, entities, stats):
    while True:
        soil_entity = yield watering_stage_queue.get()
        soil_entity.arrival_time = env.now
        soil_entity.stage = 'watering'
        stats.increment_entities()
        env.process(watering_stage(env, soil_entity, compact_stage_queue, service_stages['watering'], stats))

def compact_stage_process(env, compact_stage_queue,fill_stage_queue, service_stages, entities, stats):
    while True:
        soil_entity = yield compact_stage_queue.get()
        soil_entity.arrival_time = env.now
        soil_entity.stage = 'compact'
        stats.increment_entities()
        env.process(compact_stage(env, soil_entity, fill_stage_queue, service_stages['compact'], stats))



# 设置仿真环境
env = simpy.Environment()
fill_stage_queue = simpy.Store(env)
level_stage_queue = simpy.Store(env)
watering_stage_queue = simpy.Store(env)
compact_stage_queue = simpy.Store(env)

# 创建统计实例
stats = Statistics()

# 创建服务台状态
service_stages = {
    'fill': ServiceStage(env),
    'level': ServiceStage(env),
    'watering': ServiceStage(env),
    'compact': ServiceStage(env)
}

# 创建土层实体并加入到填筑阶段队列
for i in range(5):
    entity = SoilEntity(f"土层{i}",1)
    fill_stage_queue.put(entity)

# 启动土层处理过程
env.process(fill_stage_process(env, fill_stage_queue, level_stage_queue, service_stages, entities, stats))
env.process(level_stage_process(env, level_stage_queue, watering_stage_queue, service_stages, entities, stats))
env.process(watering_stage_process(env, watering_stage_queue, compact_stage_queue, service_stages, entities, stats))
env.process(compact_stage_process(env, compact_stage_queue, fill_stage_queue,service_stages, entities, stats))

# 动态展示仿真过程的统计信息
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置中文字体为黑体
# 动态展示仿真过程的统计信息
def monitor(env, stats):
    time_values = []
    average_wait_time_values = []

    fig, ax = plt.subplots(figsize=(10, 6))

    while True:
        # 每隔一段时间记录统计信息
        yield env.timeout(100)
        time_values.append(env.now)
        average_wait_time = stats.average_wait_time()
        average_wait_time_values.append(average_wait_time)

        # 更新图表
        ax.clear()
        ax.plot(time_values, average_wait_time_values, 'b-',color='blue', label='mean residence time', linewidth=2)
        ax.set_xlabel('Time/min', fontname='Times New Roman', fontsize=20)  # 设置 x 轴标签字体和大小
        ax.set_ylabel('mean residence time/min', fontname='Times New Roman', fontsize=20)  # 设置 y 轴标签字体和大小
        ax.set_title('Entity average residence time', fontname='Times New Roman', fontsize=20)  # 设置标题字体和大小
        ax.legend()
        ax.grid(True)

        # 设置坐标轴刻度字体为 Times New Roman
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)

        # 设置坐标轴数字字体为 Times New Roman
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y)}'))

        plt.pause(0.01)

        # 保存图表
        fig.savefig('average_wait_time.png')

# 启动监控过程
env.process(monitor(env, stats))

# 运行仿真
env.run(until=60000)
plt.savefig('平均滞留时间.png')

