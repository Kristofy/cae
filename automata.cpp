#include <SDL2/SDL.h>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <format>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <thread>
#include <vector>

class FpsCounter {
public:
  FpsCounter(uint32_t time_us)
      : last_time(time_us), last_timer_reset(time_us) {}

  float GetFps() const {
    return 1000.0f * 1000.0f * (float)sample_size / (float)sum;
  }

  void NextFrame(uint32_t time_us) {
    uint32_t dt = time_us - last_time;
    sum -= frame_times[rolling_counter];
    frame_times[rolling_counter] = dt;
    sum += frame_times[rolling_counter];
    rolling_counter =
        (rolling_counter + 1 >= sample_size ? 0 : rolling_counter + 1);
    last_time = time_us;
  }

  void Restart(uint32_t time_us) { last_timer_reset = time_us; }

  float GetLastDtMs() const { return (last_time - last_timer_reset) / 1000.0f; }

private:
  static constexpr uint32_t sample_size = 100;
  uint32_t last_time;
  uint32_t rolling_counter = 0;
  uint32_t frame_times[sample_size] = {0};
  uint32_t sum = 0;
  uint32_t last_timer_reset = 0;
};

class ThreadPool {
public:
  explicit ThreadPool(size_t num_threads)
      : num_threads(num_threads), task_counter(0), stop(false) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers.emplace_back([this]() {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->queue_condition.wait(
                lock, [this]() { return this->stop || !this->tasks.empty(); });
            if (this->stop && this->tasks.empty()) {
              return;
            }
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }
          task();
          {
            std::unique_lock<std::mutex> lock(this->task_counter_mutex);
            this->task_counter--;
            this->counter_condition.notify_one();
          }
        }
      });
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    queue_condition.notify_all();
    for (std::thread &worker : workers) {
      worker.join();
    }
  }

  template <class F, class... Args> void Enqueue(F &&func, Args &&...args) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      tasks.emplace([func, args...]() { func(args...); });
      queue_condition.notify_one();
    }

    {
      std::unique_lock<std::mutex> lock(task_counter_mutex);
      this->task_counter++;
    }
  }

  void Wait() {
    std::unique_lock<std::mutex> lock(task_counter_mutex);
    counter_condition.wait(lock, [this]() { return this->task_counter == 0; });
  }

  size_t GetThreadCount() const { return num_threads; }

private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::mutex task_counter_mutex;
  std::condition_variable queue_condition;
  std::condition_variable counter_condition;
  const size_t num_threads;
  size_t task_counter;
  bool stop;
};

class TransitionAssociation {
public:
  TransitionAssociation(std::initializer_list<uint32_t> number_of_neighbours,
                        uint32_t neighbour_state)
      : number_of_neighbours(std::begin(number_of_neighbours),
                             std::end(number_of_neighbours)),
        neighbour_state(neighbour_state) {}

  TransitionAssociation(uint32_t number_of_neighbour, uint32_t neighbour_state)
      : number_of_neighbours(1, number_of_neighbour),
        neighbour_state(neighbour_state) {}

  [[nodiscard]] const auto &GetNumberOfNeighbours() const noexcept {
    return number_of_neighbours;
  }
  uint32_t GetNeighbourState() const noexcept { return neighbour_state; }

private:
  std::vector<uint32_t> number_of_neighbours;
  uint32_t neighbour_state;
};

class MooreTransition {
public:
  MooreTransition(std::initializer_list<TransitionAssociation> accociations)
      : accociations(std::begin(accociations), std::end(accociations)) {}

  MooreTransition(TransitionAssociation accociation)
      : accociations(1, accociation) {}

  [[nodiscard]] const std::vector<TransitionAssociation> &
  GetAccociations() const noexcept {
    return accociations;
  }

private:
  std::vector<TransitionAssociation> accociations;
};

template <bool IsHomogenic = true> class MooreRuleSet;

template <> class MooreRuleSet<true> {
public:
  MooreRuleSet(uint32_t max_states, uint32_t default_state)
      : max_states(max_states), default_state(default_state) {}

  void AddRule(uint32_t current_state, MooreTransition transition,
               uint32_t new_state) {
    transitions.emplace_back(current_state, std::move(transition), new_state);
  }

  uint32_t GetMaxStates() const noexcept { return max_states; }
  uint32_t GetDefaultState() const noexcept { return default_state; }

  void GetMooreRulesetMax8(std::array<std::vector<uint32_t>, 8> &ruleset,
                           std::array<uint32_t, 64> &ruleset_8) const {
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> stack;
    uint32_t stack_size = 0;
    for (const auto &[current, accociation_collection, new_state] :
         transitions) {
      auto &added_rules = ruleset[current];

      const uint32_t n = accociation_collection.GetAccociations().size();
      const auto &accociations = accociation_collection.GetAccociations();

      uint32_t new_size = 0;
      for (int i = 0; i < (int)accociations.size(); i++) {
        uint32_t mx = 0;
        for (uint32_t x : accociations[i].GetNumberOfNeighbours()) {
          mx = (x > mx ? x : mx);
        }
        new_size += mx;
      }

      stack.resize(new_size);
      stack_size = 1;
      stack[0] = {0, 0, 0};

      while (stack_size != 0) {
        auto [i, active_rule, sum] = stack[--stack_size];

        if (i == n) {
          if (sum == 8) {
            added_rules.push_back(active_rule | (1 << new_state));
          }
          continue;
        }

        uint32_t neighbour_state = accociations[i].GetNeighbourState();
        for (uint32_t number_of_neighbours :
             accociations[i].GetNumberOfNeighbours()) {
          if (number_of_neighbours == 8) {
            if (sum == 0) {
              ruleset_8[current * 8 + neighbour_state] = 1 << new_state;
            }
            continue;
          }
          uint32_t new_rule =
              active_rule | (number_of_neighbours << (neighbour_state * 3 + 8));
          stack[stack_size++] = {i + 1, new_rule, sum + number_of_neighbours};
        }
      }
    }
  }

private:
  uint32_t max_states;
  uint32_t default_state;
  std::vector<std::tuple<uint32_t, MooreTransition, uint32_t>> transitions;
};

// 0 for no multithreading
template <uint32_t MaxNumberOfThreads = 0> class MooreCellularAutomataMax8 {
public:
  using byte = char;

#pragma region Predefined
  constexpr static const byte *Rocket =
      ".........................................O.................."
      "........................................OOO................."
      ".......................................OO.O.....O..........."
      ".......................................OOO.....OOO.........."
      "........................................OO....O..OO...OOO..."
      "..............................................OOO....O..O..."
      "........................................................O..."
      "........................................................O..."
      "........................................................O..."
      "........................................OOO............O...."
      "........................................O..O................"
      "........................................O..................."
      "........................................O..................."
      ".........................................O.................."
      "............................................................"
      "............................................................"
      "............................................................"
      "............................................................"
      "............................................................"
      "............................................................"
      "......................................OOO..................."
      "......................................O..O...........O......"
      "......................................O.............OOO....."
      "......................................O............OO.O....."
      "......................................O............OOO......"
      ".......................................O............OO......"
      "............................................................"
      "............................................................"
      "...................................OOO......................"
      "..................................OOOOO....................."
      "..................................OOO.OO.......OO.....O..O.."
      ".....................................OO.......OOOO........O."
      "..............................................OO.OO...O...O."
      "................................................OO.....OOOO."
      "............................................................"
      "............................................................"
      "....................O......................................."
      ".....................O......................................"
      ".OO.............O....O................................OOO..."
      "OOOO.............OOOOO..................................O..."
      "OO.OO...................................................O..."
      "..OO...................................................O...."
      "....................................O......................."
      ".....................................O......................"
      ".....................OO..........O...O......................"
      "......................OO..........OOOO...............OO....."
      ".....................OO...........................OOO.OO...."
      ".....................O............................OOOOO....."
      "...................................................OOO......"
      "............................................................"
      "......................OO...................................."
      ".............OOOO....OOOO..................................."
      "............O...O....OO.OO.................................."
      ".OOOOO..........O......OO..................................."
      "O....O.........O............................................"
      ".....O......................................................"
      "....O.......................................................";
#pragma region end

  MooreCellularAutomataMax8(uint32_t n, uint32_t m, uint32_t num_threads = 0)
      : n(n), m(m), n_states(0), matrix(new byte[n * m]),
        next(new byte[n * m]) {
    if (n == 0) {
      throw std::invalid_argument("Width must be positive.");
    }
    if (m == 0) {
      throw std::invalid_argument("Width must be positive.");
    }
    if (!matrix) {
      throw std::runtime_error("Could not reserve memory of " +
                               std::to_string(n * m) + " MiB");
    }
    if (!next) {
      throw std::runtime_error("Could not reserve memory of " +
                               std::to_string(n * m) + " MiB");
    }
    if ((MaxNumberOfThreads < 2 || num_threads < 2) &&
        MaxNumberOfThreads != 0) {
      throw std::invalid_argument(
          "If multitreading is specified then it should run with > 2 threads");
    }
    if (MaxNumberOfThreads < num_threads) {
      throw std::invalid_argument("The maximum number of threads " +
                                  std::to_string(MaxNumberOfThreads) +
                                  " is smaller than the specifed threads " +
                                  std::to_string(num_threads));
    }
    if constexpr (MaxNumberOfThreads) {
      tp = new ThreadPool(num_threads);
    }
    memset(matrix, 0, n * m);
  }

  ~MooreCellularAutomataMax8() {
    delete[] matrix;
    delete[] next;
    if constexpr (MaxNumberOfThreads) {
      delete tp;
    }
  }

  void AddRuleSet(const MooreRuleSet<true> &moore_ruleset) {
    std::fill(std::begin(ruleset_8), std::end(ruleset_8), 0);
    for (auto &xs : ruleset)
      xs.resize(0);
    moore_ruleset.GetMooreRulesetMax8(ruleset, ruleset_8);
    default_state = 1 << moore_ruleset.GetDefaultState();
    n_states = moore_ruleset.GetMaxStates();
    for (int i = 0; i < (int)(n * m); i++) {
      if (matrix[i] == 0) {
        matrix[i] = default_state;
      }
    }
  }

  void Debug() {
    auto bin = [](uint32_t n, char *buffer) {
      for (int i = 0; i < 32; i++) {
        buffer[i] = ((n & (1U << (31 - i))) ? '1' : '0');
      }
      buffer[32] = 0;
    };
    printf("A CA of %u states\n", n_states);
    printf("The default state is: %u\n\n", default_state);
    for (int i = 0; i < (int)n_states; i++) {
      printf("For state %d the following rules apply\n", i);
      char buffer[33];
      for (int k = 0; k < (int)ruleset[i].size(); k++) {
        bin(ruleset[i][k], buffer);
        printf("    ");
        for (int j = 0; j < 8; j++) {
          printf("%c%c%c ", buffer[3 * j + 0], buffer[3 * j + 1],
                 buffer[3 * j + 2]);
        }
        printf("-> %s\n", &buffer[24]);
      }
      puts("");
    }
    puts("");
    printf("Rule 8 rules:");

    for (int i = 0; i < (int)n_states; i++) {
      for (int j = 0; j < 8; j++) {
        printf("%d - %d -> %u\n", i, j, ruleset_8[i * 8 + j]);
      }
    }
  }

  // Only for GOL test
  void AddImage(int y, int x, int w, int h, const byte *data) {
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        matrix[Ind(y + i, x + j)] = 1 << (data[i * w + j] == 'O');
      }
    }
  }

  void Tick() {
#define CreateFreq(tli, tlj, tti, ttj, tri, trj, lli, llj, rri, rrj, bli, blj, \
                   bbi, bbj, bri, brj)                                         \
  0 | (uint64_t)matrix[Ind(tli - 1, tlj - 1)] << 0 * 8 |                       \
      (uint64_t)matrix[Ind(tti - 1, ttj + 0)] << 1 * 8 |                       \
      (uint64_t)matrix[Ind(tri - 1, trj + 1)] << 2 * 8 |                       \
      (uint64_t)matrix[Ind(lli + 0, llj - 1)] << 3 * 8 |                       \
      (uint64_t)matrix[Ind(rri + 0, rrj + 1)] << 4 * 8 |                       \
      (uint64_t)matrix[Ind(bli + 1, blj - 1)] << 5 * 8 |                       \
      (uint64_t)matrix[Ind(bbi + 1, bbj + 0)] << 6 * 8 |                       \
      (uint64_t)matrix[Ind(bri + 1, brj + 1)] << 7 * 8

    if constexpr (MaxNumberOfThreads == 0) {
      for (int i = 1; i < (int)m - 1; i++) {
        for (int j = 1; j < (int)n - 1; j++) {
          next[Ind(i, j)] =
              ApplyRule(matrix[Ind(i, j)], CreateFreq(i, j, i, j, i, j, i, j, i,
                                                      j, i, j, i, j, i, j));
        }
      }
    } else {
      int jobs = tp->GetThreadCount();
      for (int t = 0; t < jobs; t++) {
        int start = 1 + t * (m - 2) / jobs;
        int end = 1 + (t + 1) * (m - 2) / jobs;

        tp->Enqueue([start = start, end = end, this] {
          for (int i = start; i < end; i++) {
            for (int j = 1; j < (int)n - 1; j++) {
              next[Ind(i, j)] = ApplyRule(
                  matrix[Ind(i, j)],
                  CreateFreq(i, j, i, j, i, j, i, j, i, j, i, j, i, j, i, j));
            }
          }
        });
      }

      tp->Wait();
    }

    for (int i = 1; i < (int)m - 1; i++) {
      const int j = n - 1;
      const int r = -1;
      next[Ind(i, 0)] =
          ApplyRule(matrix[Ind(i, 0)],
                    CreateFreq(i, n, i, 0, i, 0, i, n, i, 0, i, n, i, 0, i, 0));
      next[Ind(i, j)] =
          ApplyRule(matrix[Ind(i, j)],
                    CreateFreq(i, j, i, j, i, r, i, j, i, r, i, j, i, j, i, r));
    }

    for (int j = 1; j < (int)n - 1; j++) {
      const int i = m - 1;
      const int b = -1;
      next[Ind(0, j)] =
          ApplyRule(matrix[Ind(0, j)],
                    CreateFreq(m, j, m, j, m, j, 0, j, 0, j, 0, j, 0, j, 0, j));
      next[Ind(i, j)] =
          ApplyRule(matrix[Ind(i, j)],
                    CreateFreq(i, j, i, j, i, j, i, j, i, j, b, j, b, j, b, j));
    }

    const int j = n - 1;
    const int i = m - 1;
    const int r = -1;
    const int b = -1;
    next[Ind(0, 0)] =
        ApplyRule(matrix[Ind(0, 0)],
                  CreateFreq(m, n, m, 0, m, 0, 0, n, 0, 0, 0, n, 0, 0, 0, 0));
    next[Ind(0, j)] =
        ApplyRule(matrix[Ind(0, j)],
                  CreateFreq(m, j, m, j, m, r, 0, j, 0, r, 0, j, 0, j, 0, r));
    next[Ind(i, 0)] =
        ApplyRule(matrix[Ind(i, 0)],
                  CreateFreq(i, n, i, 0, i, 0, i, n, i, 0, b, n, b, 0, b, 0));
    next[Ind(i, j)] =
        ApplyRule(matrix[Ind(i, j)],
                  CreateFreq(i, j, i, j, i, r, i, j, i, r, b, j, b, j, b, r));

    byte *tmp = matrix;
    matrix = next;
    next = tmp;

#undef CreateFreq
  }

  [[nodiscard]] const byte *GetMatrix() const noexcept { return matrix; }
  [[nodiscard]] const uint32_t GetWidth() const noexcept { return n; }
  [[nodiscard]] const uint32_t GetHeight() const noexcept { return m; }

private:
  inline byte ApplyRule(byte curr, uint64_t freq) {
    constexpr static uint32_t M = ~0xFFU;
    const auto &to_check = ruleset[ToIndex(curr)];
    const uint32_t mask = ToMask(freq);
    if (mask < 8) {
      byte res = ruleset_8[ToIndex(curr) * 8 + mask];
      return (res ? res : default_state);
    }

    for (int i = 0; i < (int)to_check.size(); i++) {
      if ((to_check[i] & M) == mask) {
        return to_check[i] & 0xFF;
      }
    }

    return default_state;
  }

  inline int Ind(int i, int j) const { return i * n + j; }

  constexpr inline uint32_t ToIndex(byte curr) {
    return __builtin_ctz((uint32_t)curr);
  }

  // The next line results in 5x speedup
  __attribute__((target("popcnt"))) constexpr inline uint32_t
  ToMask(uint64_t freq) {
    uint64_t masked0 = freq & 0x0101010101010101;
    uint64_t masked1 = freq & 0x0202020202020202;
    uint64_t masked2 = freq & 0x0404040404040404;
    uint64_t masked3 = freq & 0x0808080808080808;
    uint64_t masked4 = freq & 0x1010101010101010;
    uint64_t masked5 = freq & 0x2020202020202020;
    uint64_t masked6 = freq & 0x4040404040404040;
    uint64_t masked7 = freq & 0x8080808080808080;
    uint32_t neighbour0 = __builtin_popcountll(masked0);
    uint32_t neighbour1 = __builtin_popcountll(masked1);
    uint32_t neighbour2 = __builtin_popcountll(masked2);
    uint32_t neighbour3 = __builtin_popcountll(masked3);
    uint32_t neighbour4 = __builtin_popcountll(masked4);
    uint32_t neighbour5 = __builtin_popcountll(masked5);
    uint32_t neighbour6 = __builtin_popcountll(masked6);
    uint32_t neighbour7 = __builtin_popcountll(masked7);

    uint32_t test = neighbour0 | neighbour1 | neighbour2 | neighbour3 |
                    neighbour4 | neighbour5 | neighbour6 | neighbour7;

    if (test == 8) {
      if (neighbour0 == 8) {
        return 0;
      }
      if (neighbour1 == 8) {
        return 1;
      }
      if (neighbour2 == 8) {
        return 2;
      }
      if (neighbour3 == 8) {
        return 3;
      }
      if (neighbour4 == 8) {
        return 4;
      }
      if (neighbour5 == 8) {
        return 5;
      }
      if (neighbour6 == 8) {
        return 6;
      }
      if (neighbour7 == 8) {
        return 7;
      }
    }

    return neighbour0 << (3 * 0 + 8) | neighbour1 << (3 * 1 + 8) |
           neighbour2 << (3 * 2 + 8) | neighbour3 << (3 * 3 + 8) |
           neighbour4 << (3 * 4 + 8) | neighbour5 << (3 * 5 + 8) |
           neighbour6 << (3 * 6 + 8) | neighbour7 << (3 * 7 + 8);
  }

  uint32_t n = 0;
  uint32_t m = 0;
  uint32_t n_states = 0;
  byte *matrix = nullptr;
  byte *next = nullptr;
  ThreadPool *tp = nullptr;
  std::array<std::vector<uint32_t>, 8> ruleset;
  std::array<uint32_t, 64> ruleset_8;
  uint32_t default_state;
};

/*
# Run on:
- Linux kernel 6.4.4-200.fc3on
- 16 GB 4777MHz ram
- Ryzen 5 5800U(8C/16T)

# Hyper parameters:
- Run over 4000 Iteration and an average time is taken
- The field is 640 x 346

# Benchmark results:
# no pragma; O2; Single Thread    4691.98 us per iteration
# no pragma; O2; Multi Thread(2)  2666.66 us per iteration
# no pragma; O2; Multi Thread(4)  1690.10 us per iteration
# no pragma; O2; Multi Thread(8)  1147.83 us per iteration
# no pragma; O2; Multi Thread(16) 1060.04 us per iteration
# -mpopcnt ; O2; Single Thread    1517.68 us per iteration
# -mpopcnt ; O2; Multi Thread(2)  870.37  us per iteration
# -mpopcnt ; O2; Multi Thread(4)  524.95  us per iteration
# -mpopcnt ; O2; Multi Thread(8)  393.32  us per iteration
# -mpopcnt ; O2; Multi Thread(16) 380.42  us per iteration
*/

const int SIZE = 2;
const int HEIGHT = 1040;
const int WIDTH = 1920;
const int n = WIDTH / SIZE;
const int m = HEIGHT / SIZE;
constexpr int nthreads = 8;
constexpr int ITER = 4000;

int main() {
  auto now = [] {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };
  using namespace std::chrono_literals;
  MooreCellularAutomataMax8<nthreads> ca(n, m, nthreads);

  constexpr uint32_t ALIVE = 1;
  constexpr uint32_t DEAD = 0;

  // GOL
  MooreRuleSet<true> Gol(2, DEAD);
  Gol.AddRule(DEAD,
              MooreTransition{
                  {TransitionAssociation{3, ALIVE},
                   TransitionAssociation{{0, 1, 2, 3, 4, 5, 6, 7, 8}, DEAD}}},
              ALIVE);
  Gol.AddRule(ALIVE,
              MooreTransition{
                  {TransitionAssociation{{2, 3}, ALIVE},
                   TransitionAssociation{{0, 1, 2, 3, 4, 5, 6, 7, 8}, DEAD}}},
              ALIVE);
  ca.AddRuleSet(Gol);

  // Day and Night:
  // MooreRuleSet<true> DayAndNight(2, DEAD);
  // DayAndNight.AddRule(DEAD, MooreTransition{{TransitionAssociation{{3, 6, 7,
  // 8}, ALIVE}, TransitionAssociation{{0,1,2,3,4,5,6,7,8}, DEAD}} }, ALIVE);
  // DayAndNight.AddRule(ALIVE, MooreTransition{{TransitionAssociation{{3, 4, 5,
  // 6, 7, 8}, ALIVE}, TransitionAssociation{{0,1,2,3,4,5,6,7,8}, DEAD}} },
  // ALIVE); ca.AddRuleSet(DayAndNight);

  // Seeds:
  // MooreRuleSet<true> Seeds(2, DEAD);
  // Seeds.AddRule(DEAD, MooreTransition{{TransitionAssociation{{2}, ALIVE},
  // TransitionAssociation{{0,1,2,3,4,5,6,7,8}, DEAD}} }, ALIVE);
  // ca.AddRuleSet(Seeds);

  // ca.Debug();

  const char *glider = ".O."
                       "..O"
                       "OOO";

  ca.AddImage(2, 0, 60, 47, MooreCellularAutomataMax8<0>::Rocket);

  // auto start = now();
  // for(int i = 0; i < ITER; i++) {
  // 	ca.Tick();
  // }
  // auto end = now();
  // FILE* f = fopen("/tmp/benchmark_result", "w");
  // fprintf(f, "Code took %.2f ms in total\n", (end - start) / 1000.0f);
  // fprintf(f, "%.2f us per iteration\n", (end - start) / (float)ITER);
  // fclose(f);

#if 0
  char buff[128];
  char memory[n * m];

  memset(memory, 0, n * m);

  SDL_Init(SDL_INIT_EVERYTHING);

  SDL_Window *window =
      SDL_CreateWindow("hac", 0, 0, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
  SDL_Event event;
  bool running = true;

  SDL_Renderer *renderer =
      SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

  float target_fps = 30000.0f;
  FpsCounter fps_counter(now());
  while (running) {
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        running = false;
        break;
      }
    }

    SDL_SetRenderDrawColor(renderer, 51, 51, 51, 255);
    SDL_RenderFillRect(renderer, NULL);

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        char mem = memory[i * n + j];
        const int red_level[] = {0, 50, 75};
        const int green_level[] = {0, 60, 140, 230};
        const int blue_level[] = {0, 60, 140, 230};
        int r_n = __builtin_popcount(0b00011000 & mem);
        int g_n = __builtin_popcount(0b00000111 & mem);
        int b_n = __builtin_popcount(0b11100000 & mem);
        int len = __builtin_popcount(mem);

        SDL_Rect rect;
        int len_level = (len > 6) + (len > 4) + (len > 2) + (len > 1);
        SDL_SetRenderDrawColor(renderer, red_level[r_n], green_level[g_n],
                               blue_level[b_n], 255);
        switch (len_level) {
        case 0: {
          rect.y = i * SIZE + 1;
          rect.x = j * SIZE - 1;
          rect.w = SIZE - 3;
          rect.h = SIZE - 3;
        } break;
        case 1: {
          rect.y = i * SIZE + 1;
          rect.x = j * SIZE - 1;
          rect.w = SIZE - 2;
          rect.h = SIZE - 2;
        } break;
        case 2: {
          int offset_x = 2 * (rand() / RAND_MAX < 0.5f) - 1;
          int offset_y = offset_x - 1;
          rect.y = i * SIZE + offset_y;
          rect.x = j * SIZE - offset_x;
          rect.w = SIZE - offset_x;
          rect.h = SIZE - offset_y;
        } break;
        case 3: {
          int offset_x = 2 * (rand() / RAND_MAX < 0.5f) - 1;
          int offset_y = offset_x - 1;
          rect.y = i * SIZE + offset_y;
          rect.x = j * SIZE - offset_x;
          rect.w = SIZE - 1;
          rect.h = SIZE - 1;
        } break;
        case 4: {
          rect.y = i * SIZE;
          rect.x = j * SIZE;
          rect.w = SIZE;
          rect.h = SIZE;
        } break;
        }
        SDL_RenderFillRect(renderer, &rect);
      }
    }
    for (int i = 0; i < 100; i++)
      ca.Tick();
    for (int i = 0; i < n * m; i++) {
      memory[i] <<= 1;
      memory[i] |= (ca.GetMatrix()[i] == ALIVE);
    }

    SDL_RenderPresent(renderer);

    sprintf(buff, "HCA: FPS: %f", fps_counter.GetFps());
    SDL_SetWindowTitle(window, buff);

    fps_counter.NextFrame(now());
    auto ms = fps_counter.GetLastDtMs();
    if (ms < 1000.0f / target_fps) {
      std::this_thread::sleep_for((1000.0f / target_fps - ms) * 1ms);
    }
    fps_counter.Restart(now());
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
#else

  for(int i = 0; i < 10'000; i++)
    ca.Tick();

  float target_fps = 1.0f;
  FpsCounter fps_counter(now());
  while (true) {
    auto *matrix = ca.GetMatrix();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        int index = i * n + j;
        printf("%c", (matrix[index] == 2 ? '#' : '.'));
      }
      puts("");
    }
    printf("\n\n\n");

    // ca.Tick();

    // fps_counter.NextFrame(now());
    // auto ms = fps_counter.GetLastDtMs();
    // if (ms < 1000.0f / target_fps) {
    //   std::this_thread::sleep_for((1000.0f / target_fps - ms) * 1ms);
    // }
    // fps_counter.Restart(now());
    break;
  }
#endif

  return 0;
}
