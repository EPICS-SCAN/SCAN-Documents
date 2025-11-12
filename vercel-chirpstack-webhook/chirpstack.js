const admin = require("firebase-admin");
const serviceAccount = require("../serviceAccountKey.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const db = admin.firestore();

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    return res.status(405).send("Method Not Allowed");
  }

  const data = req.body;

  // Log full incoming payload for debugging
  console.log("Incoming ChirpStack payload:", JSON.stringify(data, null, 2));

  let decodedValue = null;
  if (data.data) {
    try {
      const buffer = Buffer.from(data.data, "base64");
      const asciiString = buffer.toString("utf8"); // "0057"
      decodedValue = parseInt(asciiString, 10); // Convert to number (57)
    } catch (error) {
      console.error("Error decoding base64:", error);
    }
  }

  try {
    await db.collection("garages").doc("temp").set(
      {
        lastSeen: new Date(),
        available: decodedValue || 0 // Store as number
      },
      { merge: true }
    );

    res.status(200).send("Data stored in garages/temp");
  } catch (error) {
    console.error("Error writing to Firestore:", error);
    res.status(500).send("Failed to store data");
  }
};