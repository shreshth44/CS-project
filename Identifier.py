import numpy as np
import pygame
import sys
import tensorflow as tf
import time

model = tf.keras.models.load_model('model.h5')

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

pygame.init()
width, height = 600, 400
size = width, height
screen = pygame.display.set_mode(size)
pygame.display.set_caption('Drawing Identifier')

OPEN_SANS = "OpenSans-Regular.ttf"
smallFont = pygame.font.Font(OPEN_SANS, 20)
largeFont = pygame.font.Font(OPEN_SANS, 40)

ROWS, COLS = 28, 28

OFFSET = 20
CELL_SIZE = 10

handwriting = [[0] * COLS for _ in range(ROWS)]
classification = None

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(BLACK)

    click, _, _ = pygame.mouse.get_pressed()
    if click == 1:
        mouse = pygame.mouse.get_pos()
    else:
        mouse = None

    cells = []
    for i in range(ROWS):
        row = []
        for j in range(COLS):
            rect = pygame.Rect(
                OFFSET + j * CELL_SIZE,
                OFFSET + i * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )

            # darken cell
            if handwriting[i][j]:
                channel = 255 - (handwriting[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)

            # blank cell
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1) # boundary of cell

            # If writing on this cell, fill in current cell and neighbors
            if mouse and rect.collidepoint(mouse):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

    # reset button
    resetButton = pygame.Rect(
        30, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    resetText = smallFont.render("Reset", True, BLACK)
    resetTextRect = resetText.get_rect()
    resetTextRect.center = resetButton.center
    pygame.draw.rect(screen, WHITE, resetButton)
    screen.blit(resetText, resetTextRect)

    # classify button
    classifyButton = pygame.Rect(
        150, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    classifyText = smallFont.render("Classify", True, BLACK)
    classifyTextRect = classifyText.get_rect()
    classifyTextRect.center = classifyButton.center
    pygame.draw.rect(screen, WHITE, classifyButton)
    screen.blit(classifyText, classifyTextRect)

    # reset drawing
    if mouse and resetButton.collidepoint(mouse):
        handwriting = [[0] * COLS for _ in range(ROWS)]
        classification = None

    # classify
    if mouse and classifyButton.collidepoint(mouse):
        classification = model.predict(
            [np.array(handwriting).reshape((1, 28, 28, 1))]
        )
        percentage = classification.max()
        classification=classification.argmax()

    # show classification
    if classification is not None:
        classificationText = largeFont.render(str(classification), True, WHITE)
        classificationRect = classificationText.get_rect()
        grid_size = OFFSET * 2 + CELL_SIZE * COLS
        classificationRect.center = (
            grid_size + ((width - grid_size) / 2),
            100
        )
        percentageText = smallFont.render(f'{round(percentage*100, 2)}% sure', True, WHITE)
        perRect = percentageText.get_rect()
        perRect.center=(classificationRect.center[0], 150)

        screen.blit(classificationText, classificationRect)
        screen.blit(percentageText, perRect)

    pygame.display.flip()
