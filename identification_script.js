const express = require('express');
const multer = require('multer');
const path = require('path');
const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const { createCanvas, loadImage } = require('canvas');

const app = express();
const port = 3000;

// Set up storage for uploaded images
const storage = multer.diskStorage({
  destination: './uploads',
  filename: (req, file, cb) => {
    cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
  },
});

// Set up view engine
app.set('view engine', 'ejs');

// Serve static files from the 'public' directory
app.use(express.static('public'));

// Home route
app.get('/', (req, res) => {
  res.render('index');
});

// Upload route
app.post('/upload', async (req, res) => {
  try {
    await upload(req, res);

    if (req.file) {
      const imagePath = path.resolve(__dirname, req.file.path);

      // Call the identifyAnimal function and wait for the result
      const predictions = getHighestProbability(await identifyAnimal(imagePath));

      // Render the image with text and predictions
      res.render('display', { predictions });
    } else {
      res.status(400).send('No image file provided.');
    }
  } catch (err) {
    console.error(err);
    res.status(500).send(err.message || 'Internal Server Error');
  }
});

// Modified upload middleware to handle errors
const upload = (req, res) => {
  return new Promise((resolve, reject) => {
    const uploadMiddleware = multer({
      storage: storage,
    }).single('image');

    uploadMiddleware(req, res, (err) => {
      if (err) {
        reject(err);
      } else {
        resolve();
      }
    });
  });
};

// Function to get the prediction with the highest probability
function getHighestProbability(predictions) {
  if (!predictions || predictions.length === 0) {
    return null;
  }

  return predictions.reduce((max, prediction) => (
    prediction.probability > max.probability ? prediction : max
  ), predictions[0]);
}

async function identifyAnimal(imagePath) {
  const image = await loadImage(imagePath);
  const canvas = createCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, image.width, image.height);

  const input = tf.browser.fromPixels(canvas);
  const model = await mobilenet.load();

  const predictions = await model.classify(input);
  console.log('Predictions:', predictions);

  return predictions;
}

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
