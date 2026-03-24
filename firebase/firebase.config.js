// File: MediGraph/firebase/firebase.config.js
// Path: /Users/shweta/Desktop/major project fy/MediGraph/firebase/firebase.config.js

/**
 * Firebase Configuration for MediGraph
 * 
 * IMPORTANT: Replace placeholder values with your actual Firebase project credentials.
 * These values can be found in Firebase Console > Project Settings > Your apps > SDK setup
 */

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID
};

export default firebaseConfig;
