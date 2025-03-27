import random
import matplotlib.pyplot as plt


def rps_winner(player_move, computer_move):
    if player_move == computer_move:
        return 0
    if player_move == (computer_move + 1) % 3:
        return 1
    return -1


def predict_next_move(last_move, transition_counts):
    if last_move is None:
        return random.choice([0, 1, 2])

    row_counts = transition_counts[last_move]
    row_sum = sum(row_counts)
    if row_sum == 0:
        return random.choice([0, 1, 2])

    probabilities = [count / row_sum for count in row_counts]
    rand_val = random.random()
    cumulative = 0.0
    for move_index, prob in enumerate(probabilities):
        cumulative += prob
        if rand_val < cumulative:
            return move_index
    return 2


def choose_computer_move(predicted_player_move):
    return (predicted_player_move + 1) % 3


def update_transition_counts(transition_counts, old_move, new_move):
    if old_move is not None:
        transition_counts[old_move][new_move] += 1


def fake_player_decide_move(last_player_move, last_computer_move, last_winner, stick_probability=0.65,
                            switch_probability=0.65):
    if last_player_move is None or last_computer_move is None or last_winner is None:
        return random.choice([0, 1, 2])

    if last_winner == "tie" or last_winner == "computer":
        if random.random() < switch_probability:
            return (last_player_move + 2) % 3
        else:
            return random.choice([0, 1, 2])

    if last_winner == "player":
        if random.random() < stick_probability:
            return last_computer_move
        else:
            return random.choice([0, 1, 2])


def train_computer(transition_counts, num_simulated_games=64):
    last_player_move = None
    last_computer_move = None

    player_score = 0
    computer_score = 0

    last_winner = None

    advantage_history = []

    for _ in range(num_simulated_games):
        predicted_player_move = predict_next_move(last_player_move, transition_counts)
        computer_move = choose_computer_move(predicted_player_move)

        player_move = fake_player_decide_move(last_player_move, last_computer_move, last_winner)

        update_transition_counts(transition_counts, last_player_move, player_move)

        result = rps_winner(player_move, computer_move)
        if result == 1:
            player_score += 1
            last_winner = "player"
        elif result == -1:
            computer_score += 1
            last_winner = "computer"
        else:
            last_winner = "tie"

        advantage_history.append(player_score - computer_score)

        last_player_move = player_move
        last_computer_move = computer_move

    print(f"Computer trained on {num_simulated_games} simulated games.")
    print(f"Fake Player: {player_score}, Computer: {computer_score}")
    print(f"Last training round winner: {last_winner}")

    rounds = range(1, len(advantage_history) + 1)
    plt.plot(rounds, advantage_history)
    plt.title("Fake Player's Score Advantage (Training Phase)")
    plt.xlabel("Training Round")
    plt.ylabel("Advantage (Fake Player - Computer)")
    plt.show()


def move_to_index(move_str):
    m = move_str.strip().upper()
    if m.startswith('R'):
        return 0
    if m.startswith('P'):
        return 1
    if m.startswith('S'):
        return 2
    return None


def main():
    transition_counts = [
        [0, 0, 0],  # rock
        [0, 0, 0],  # paper
        [0, 0, 0]  # scissors
    ]

    train_computer(transition_counts, num_simulated_games=64)

    target_score = 5
    player_score = 0
    computer_score = 0

    last_player_move = None
    round_number = 0

    print("\nWelcome to Rock-Paper-Scissors!")
    print("Enter 'R' for Rock, 'P' for Paper, 'S' for Scissors.")
    print(f"First to {target_score} points wins.\n")

    while player_score < target_score and computer_score < target_score:
        round_number += 1

        predicted_player_move = predict_next_move(last_player_move, transition_counts)
        computer_move = choose_computer_move(predicted_player_move)

        player_input = input(f"Round {round_number} - Your move (R/P/S): ")
        player_move = move_to_index(player_input)

        if player_move is None:
            print("Invalid move, please enter R, P, or S.\n")
            round_number -= 1
            continue

        update_transition_counts(transition_counts, last_player_move, player_move)
        last_player_move = player_move

        result = rps_winner(player_move, computer_move)
        if result == 1:
            player_score += 1
            print("You won this round!")
        elif result == -1:
            computer_score += 1
            print("Computer won this round!")
        else:
            print("It's a tie!")

        print(f"Player move: {['Rock', 'Paper', 'Scissors'][player_move]}, "
              f"Computer move: {['Rock', 'Paper', 'Scissors'][computer_move]}")
        print(f"Score => You: {player_score} | Computer: {computer_score}\n")

    if player_score > computer_score:
        print("You win the match!")
    else:
        print("Computer wins the match!")


if __name__ == "__main__":
    main()
