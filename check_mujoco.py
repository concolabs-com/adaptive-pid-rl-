import gymnasium as gym


def check_mujoco_access():
    print("Checking MuJoCo access...")
    try:
        # Using InvertedPendulum-v4 which is standard in gymnasium > 0.26
        env = gym.make("InvertedPendulum-v4", render_mode=None)
        unwrapped = env.unwrapped

        # Access through .model (MuJoCo object)
        model = unwrapped.model

        # Modify body mass (index 1 is likely the pole)
        print(f"Model body_mass before: {model.body_mass[1]}")
        model.body_mass[1] *= 2.0
        print(f"Model body_mass after: {model.body_mass[1]}")

        # Modify geom friction (index 0 is likely the rail/cart interaction)
        # geom_friction shape is (ngeom, 3) usually [sliding, torsional, rolling]
        print(f"Geom friction before: {model.geom_friction[0]}")
        model.geom_friction[0, 0] *= 0.5
        print(f"Geom friction after: {model.geom_friction[0]}")

        env.reset()
        print("Reset successful. MuJoCo is working and modifiable.")

    except Exception as e:
        print(f"Error accessing MuJoCo environment: {e}")


if __name__ == "__main__":
    check_mujoco_access()
