# Log

**11.08.2026**
- ресёрч mjbots/moteus: компания из США (Cambridge, MA, Josh Pieper, 2019), всё open-source (Apache 2.0). Интерфейс CAN-FD 5 Mbps, библиотеки Python/C++/Rust, GUI tview. Рекомендуемый стек: mj5208 + moteus-c1 + Raspberry Pi + pi3hat (CAN-FD + IMU на борту) → [hardware.md → Driver: moteus](docs/hardware/hardware.md)
- ⚠️ mjbots: shipping holiday, новые заказы отправляются с 17.08.2026

**10.08.2026 (4)**
- **решение пересмотрено: мотор = mjbots mj5208 ($74) + драйвер moteus** ([mj5208](https://mjbots.com/products/mj5208), [moteus-c1 $69](https://mjbots.com/products/moteus-c1) / [r4.11 $94](https://mjbots.com/products/moteus-r4-11)). Причина: отказ от SimpleFOC снял лимит 5 A — mj5208 (Kv 330, K_t≈0.029) с c1 даёт 0.58 Nm peak, с r4.11 до 1.7 Nm при 193 g (8.8 Nm/kg — лучший specific torque из всех), ω_nl@4S ≈ 4900 rpm — momentum перестаёт быть проблемой. Motor+driver+encoder ≈ $143 — та же цена, что GL40+c1. Devkit-пара: плата ставится на задний магнит ротора, энкодер на борту
- открытые вопросы перед заказом: cogging mj5208 (не gimbal-обмотка; у moteus есть anti-cogging — проверить на форуме/discord mjbots); доставка из США (+~25–30% VAT/пошлина, сроки); continuous torque на c1 без обдува всего 0.14 Nm — вероятно хватает для удержания, иначе r4.11
- GL40 KV70 остаётся запасным вариантом (EU-склад, быстрая доставка)

**10.08.2026 (3)**
- **решение: мотор = CubeMars GL40 KV70** (вариант с энкодером, [OpenELAB €110](https://openelab.io/products/cubemars-gl40-kv70-bldc-gimbal)). Обоснование: лучший specific torque (4.6 Nm/kg), 107 g, 940 rpm @ 4S. Коммиты решения: 4S обязателен (на 3S лишь 0.37 Nm); потолок 0.49 Nm — запас k≈1.5–1.9 при θ_max=20° с батареей на оси, Phase 0 подтверждает реальными массами; перед заказом уточнить тип/интерфейс энкодера у OpenELAB (не раскрыт) или взять base + AS5048A; болтовое поле ротора — из STEP-чертежа CubeMars
- покупка НЕ оформлена — ждёт Phase 0 gate

**10.08.2026 (2)**
- решение: батарея на оси вращения → почти не добавляет m·l; порог момента снижен до ~0.45 Nm @ 4S — у GL40 KV70 появился реальный запас
- swing-up: разгон-реверс колеса = стандартный energy pumping; сила «кика» ограничена K_t·I_max независимо от RPM — метрика — momentum capacity I_w·ω, добирать инерцией колеса и многотактовой раскачкой, не высоким Kv
- второй ресёрч: +8 новых кандидатов (SteadyWin, MyActuator/LK-TECH) → top-7 buy list с URL в [motor-candidates.md](docs/hardware/motor-candidates.md); новый класс: сервоприводы со встроенным FOC-драйвером и энкодером (CAN) — RMD-L-5015 (0.82 Nm, 174 g, ~€72)

**10.08.2026**
- сформулированы критерии выбора BLDC-мотора ([hardware.md](docs/hardware/hardware.md#motor-selection-criteria)): parametric torque budget, desktop DIY-kit, батарея на борту (3S–4S), мотор ≤ €120
- ресёрч кандидатов завершён: 15 моторов оценено → [motor-candidates.md](docs/hardware/motor-candidates.md). Shortlist: CubeMars GL40 KV70 (€85–110, 0.49 Nm@4S — ровно на пороге), iPower GM5208-24 (€52, риск по swing-up), CubeMars G60 KV55 (€132, 1.0 Nm — stretch). Выводы: 4S обязателен; в бюджете рынок упирается в ~0.5 Nm; драйвер нужен ≥5 A FOC

**09.08.2026**
Проект воскрешён: перенесён из `archive/projects/` обратно в `projects/`. Состояние на момент заморозки: в симуляции всё работает (SB3/PPO, swing-up, ONNX + браузерная визуализация), железо упёрлось в узкий диапазон рабочих параметров (момент двигателя, точность измерений). LQR посчитан, но колесо не останавливалось.

**01.11.2025**
Сегодня я решил похоронить обратный маятник. Все очень здорово выглядит в симуляторе, пока не нужно проектировать реальное устройство. Легко стабилизировать, когда у тебя измерения произвольной точности и бесконечный момент двигателя. В реальности, есть очень узкий набор параметров, при котором баланс достижим. Этот проект уже занял у меня кучу времени, и я не хочу тратить на него свою жизнь. Он, как настоящий маятник из трансерфинга реальности, просто оттягивает мою жизненную энергию на себя, больше я не буду его кормить. Это был проект, над которым у меня было 100% контроля и 0% пользы для меня или для мира.

**05.10.2025**
- estimated maximal angular velocity, it corresponds to the reality
- fixed jerky angle at the top, now it is smooth
- calculated LQR stabilization (but the wheel does not stop), needs more tuning, or since it does not have a concept of time (no integral part) there is no sufficient information for control
- → next: stabilize in the upright position using LQR

**03.10.2025**
- implemented active oscillation dumping, [simulation](https://colab.research.google.com/drive/1u6tl5SG2cvKg8DLndMQ9u2ieP07KyDuc#scrollTo=PbFrm5GsAxk4)
- dumping force is just $-\dot{\theta}$

**07.09.2025**
- switched back to MPU6050, since I need only roll
- [x] redesign MPU position to rotate around X-axis (ordered a new board)
- used https://github.com/TKJElectronics/KalmanFilter/tree/master library to get roll without a drift
- [x] refactor code, incapsulate MPU-related code for improved readability

**20.05.2025**
- used `kwokkayan/SparkFun MPU-9250 Digital Motion Processing (DMP) Arduino Library (For ESP32)@^1.0.1` library for IMU, it worked, didn't trigger a reset cycle
- only `roll` is a reliable reading now
- → 🔥 find a reliable value for an angle in the vertical plane (use 6DOF and implement sensor fusion)
- → try a lighter motor, see if it has enough torque to rotate the wheel
- → 🔥 redesign a rod, add mounting wholes for the PCB
- → stabilize the pendulum in the bottom position (with PID, then NN)
- => feels like I want to abandon using Gyro... and get back to the design with an angular encoder
	- try 6DOF algorithms so far

**17.05.2025**
- → select a bearing with axial load tolerance
- → fix IMU init issue, previous (balancing robot) firmware also turns into reboot (try with Arduino, then with a SparkFun WROOM board)
- → compare different IMU modules `ICM-20948`, `BMI270`, `BNO055`, `BMI088`, `BMI323`
- ~~→ забрать тиски у Димана~~
- → забрать дрель у Алины
- → уменьшить толщину стенки, где крепляется двигатель (<7mm) или найти более длинный болт M3 (>8mm)
- → review TriNamic stepper motor drivers, e.g. TMC2209
- [ch34xser_macos driver should be installed](https://github.com/WCHSoftGroup/ch34xser_macos?tab=readme-ov-file)
- stepper enabled
- issues with MPU, goes into a reboot cycle, check if the board is functional
```
clk_drv:0x00,q_drv:0x00,d_drv:0x00,cs0_drv:0x00,hd_drv:0x00,wp_drv:0x00
mode:DIO, clock div:2
load:0x3fff0030,len:1184
load:0x40078000,len:13232
load:0x40080400,len:3028
entry 0x400805e4
before MPU begin
[   647][E][Wire.cpp:499] requestFrom(): i2cWriteReadNonStop returned Error 263
[   655][E][Wire.cpp:499] requestFrom(): i2cWriteReadNonStop returned Error -1
[   662][E][Wire.cpp:499] requestFrom(): i2cWriteReadNonStop returned Error -1
[   669][E][Wire.cpp:499] requestFrom(): i2cWriteReadNonStop returned Error -1
Guru Meditation Error: Core  1 panic'ed (LoadProhibited). Exception was unhandled.
```

**29.03.2025**
- сделал визуализацию в браузере, TypeScript, стабилизировал с помощью экспортированной ONNX-модели
- → до обучить модель, чтобы она стремилась остановить маятник в верхнем положении (2х фазное обучение, разная функция вознаграждения на каждом этапе)
- ✅ сделать задачи из TODO-листа по вебу - задеплоить
- → опубликовать статью по экспорту модели в ONNX
- → опубликовать статью по проекту на данной фазе
- → dedicated landing page
- → publish articles on LinkedIn

**18.03.2025**
- `action_net` использует нормализацию с `tahn`-функцией, потом масштабирование
```python
mean_actions = self.policy.action_net(latent_pi)  
normalized_actions = torch.tanh(mean_actions)  
scaled_actions = self.action_low + (0.5 * (normalized_actions + 1.0) * (self.action_high - self.action_low))
```
- удалось успешно экспортировать модель в ONNX в один прогон, визуализировать с ней стабилизацию в исходном окружении
- ✅ визуализировать все красиво в браузере, позволить пользователю "пинать" маятник мышкой с разных сторон

**15.03.2025**
- использовал [claude.ai](https://claude.ai), он отлично и глубоко анализирует проблемы
- убедился, что архитектуры PyTorch и ONNX-моделей должны совпадать
- собрал ONNX-модель вручную, наполнил ее параметрами из PyTorch-модели, результат получился почти идентичным, если сравнивать послойно
- Но! результат прогона полной PPO-модели сильно отличается, PPO (-4, 4), тогда, как послойный запуск дает -53..53, там есть еще какая-то пост-обработка
- ✅ выяснить, что за пост-обработка в PPO-модели присутствует
- ✅ добавить ее в ONNX-модель
- ✅ сделать базовую визуализацию с ONNX-моделью

**13.03.2025**
- вчера экспортировал модель из PyTorch в ONNX, возникли такие проблемы:
	- экспортированная модель не повторяет архитектуру исходной сети, получается какой-то сложный граф
	- на выходе 3 значения (action, value, log_prob)
	- результат разный на одних и тех же данных, возможно, там появился слой drop-out
	- выходное значение на несколько порядков отличается от выходного значения исходной нейронной сети (но можно подобрать коэффициент сжатия)
- **идеи**
	- экспортировать только `actor` network
	- ✅ try to code a simple PyTorch model and export it, the architecture should be the same, it should give the same values
	- загрузить параметры PyTorch модели вручную и самому реализовать сеть
	- ✅ собрать ONNX модель самому, просто скопировав параметры из словаря PyTorch модели
	- Реализовать nn.Model архитектуру `actor` network, загрузить в нее параметры из исходной ActorCritic, экспортировать в ONNX, убедиться, что архитектура осталась прежней и значения выдаются такие же.
**23.02.2025**
- exported model to ONNX format, can be executed separately from PyTorch environment
- → render using ONNX model
- → implement forward pass in C++ (or TinyTorch)

**01.02.2025**
 - tried a new reward function: 
   $r = -(\theta^2 + 0.1\cdot\dot\theta^2 + 0.001\cdot\dot\phi^2) -  termination\_penalty$
   - it can learn how to balance from a narrow set of angle upwards
   - it can learn how to swing-up
   - but how to combine those skills?
   - → try to train one set of skills, then pre-load the model and train another set of skills, then mix conditions

**27.01.2025**
- все зависит от функции вознаграждения, в нижнем положении, он отлично научился раскачиваться
- но с умением раскачиваться, ему выгодней просто какое-то время находиться в верхней половине, над горизонтом
- → поискать функции вознаграждения у тех, кто уже делал раскачивание
- → раскачать классическую задачу с маятником на тележке
- → [try this reward function](https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py)

**26.01.2025**
Run in Kaggle notebook on GPU: seems no benefit:
- CPU: 50,000 steps Elapsed time: **114.079** seconds
- GPU: Elapsed time: **149.339** seconds
- → consider this [repo](https://github.com/0xangelo/gym-cartpole-swingup) for Swing-Up and inheritance pattern
- → render model usage phase with Manim

**23.01.2025**
- decided not to vectorize the environment for now, too long to study this topic and change the env to support it.
- run CartPole on Kaggle, checked GPU availability.
- → train to swing-up on Kaggle, reuse trained model locally for visualization
- → show training graph in Kaggle (reward change)

**21.01.2025**
- stabilized FIP with SB3, reward function was crucial, it sets accents on which aspects of the reality the model should focus, eventually it became more aggressive, with higher exploration coefficient.
- → visualize mean reward and other params during training
- → visualize NN internals, how it operates from the inside
- → next: vectorize to speedup training → CANCELLED: too long to study
- → next: swing-up, even more aggressive control

**19.01.2025**
- stabilized CartPole-v1 with SB3, the original Env had a discrete action space, I made it continuous, it made the system more robust to disturbances
- an external disturbance was added during the testing phase
- pole angle threshold was increased during the testing phase to allow bigger movements before the episode ends
- → ✅ stabilize FIP with SB3
- → train a cart-pole to swing-up, allow to apply a bigger force
- → vectorized environments with GPU @Kaggle → publish an article

**18.01.2025**
- restored stabilization with LQR
- rendered env parameters
- → ✅ stabilize cart-pole with stable baselines3
- → ✅ FIP with baselines3 (find right initial conditions)

**14.01.2025**
- `FlywheelInvertedPendulumEnv`, internal variables are not accessible due to wrapper classes, it worked before, have no idea why it stopped working
- go for FIP directly, forget about ball catcher, rather remove it
- → ✅ finish HF
- → try Cart Pole and Moon Lander locally with SB3
- → learn fast-ai course
