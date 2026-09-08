#include "football_fan_ex/football_fan_ex.hpp"

#include <SFML/Graphics.hpp>

#include <optional>

int main() {
    sf::RenderWindow window(sf::VideoMode({1920u, 1080u}), "Football Fan Ex");

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }
    }

    return 0;
}
