**To:** Prof. Yapeng Tian
**Subject:** CS 4391 project: SpotCheck scope change (Ridham Gabani, Om Patel, Krishna Chundru, Kevin Gajera)

Dear Prof. Tian,

Thank you for approving our proposal, "SpotCheck: Detecting Parking Space Occupancy on the UT Dallas Campus."

Because of time, we will not be able to collect and label our own campus photos. We'd like to keep the same
question and method but change the data:

- **New site:** instead of UTD lots, we hold out one of PKLot's three real lots (PUCPR) as the unseen "new
  site." We train on the other two (UFPR04, UFPR05).
- **Method (unchanged):** a COCO-pretrained YOLOv8 fine-tuned for empty/occupied spaces. We compare
  source-only, new-site-only, fine-tuned, and combined training, plus YOLOv8n vs. YOLOv8s. We also add a curve
  showing how many labeled new-site images (25 to 200) fine-tuning needs.
- **Evaluation (unchanged metrics):** mAP@0.5, mAP@0.5:0.95, per-class precision and recall, empty-space
  counting error, and FPS. Instead of day/dusk/night, the breakdown uses PKLot's real weather labels (sunny,
  cloudy, rainy) and time of day from the image timestamps.

Please let us know if this change is acceptable or if you'd like us to adjust anything.

Best regards,
Ridham Gabani, Om Patel, Krishna Chundru, Kevin Gajera
