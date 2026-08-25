"""Example: generate matched baseline and MI-like cohorts."""
from cardisim import CardiacSimulator, SimulationConfig, population_preset


def main() -> None:
    config = SimulationConfig(duration=28, dt=0.25, n_cells=256, seed=42, process_noise=0.002)
    simulator = CardiacSimulator(config)

    control = simulator.run(population_preset("baseline"))
    mi = simulator.run(population_preset("mi"))

    print("CONTROL health:", control.summary()["cardiac_health_score"])
    print("MI health:", mi.summary()["cardiac_health_score"])
    print("MI events:", mi.events)
    mi.to_csv("outputs/mi_population.csv")


if __name__ == "__main__":
    main()
