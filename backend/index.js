require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const path = require('path');
const { MongoClient } = require('mongodb');
const app = express();
const PORT = process.env.PORT;

// Middleware to serve static files from 'assets' folder
app.use("/assets", express.static(path.join(__dirname, "assets")));

// Middleware for body parsing
app.use(bodyParser.urlencoded({ extended: true }));

// Serve Login Page
app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Admin Login</title>
      <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
      <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }
        body {
          font-family: 'Poppins', sans-serif;
          display: flex;
          justify-content: center;
          align-items: center;
          height: 100vh;
          background: #000;
          color: #fff;
          overflow: hidden;
        }
        #particles-js {
          position: absolute;
          width: 100%;
          height: 100%;
          z-index: -1;
        }
        .login-container {
          text-align: center;
          background: rgba(0, 0, 0, 0.6);
          padding: 50px 30px;
          border-radius: 20px;
          box-shadow: 0 10px 20px rgba(0, 0, 0, 0.8);
          max-width: 500px;
          width: 95%;
          animation: slideIn 1s ease-out forwards;
          transition: transform 0.3s ease;
        }
        .login-container:hover {
          transform: scale(1.05);
        }
        .logo {
          width: 120px;
          height: 120px;
          margin-bottom: 20px;
          border-radius: 50%;
          border: 3px solid #fff;
          animation: pulse 1.5s infinite;
        }
        h1 {
          font-size: 36px;
          font-weight: 600;
          margin-bottom: 20px;
          color: #fff;
          text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
        }
        input {
          width: 85%;
          padding: 12px 15px;
          margin: 15px 0;
          border: 2px solid #fff;
          border-radius: 5px;
          background: rgba(255, 255, 255, 0.2);
          color: #fff;
          font-size: 16px;
          transition: all 0.3s ease;
        }
        input:focus {
          border-color: #4CAF50;
          background: rgba(255, 255, 255, 0.1);
          outline: none;
        }
        button {
          width: 90%;
          padding: 12px 20px;
          border: none;
          border-radius: 5px;
          background-color: #4CAF50;
          color: #fff;
          font-size: 18px;
          cursor: pointer;
          margin-top: 20px;
          transition: all 0.3s ease;
        }
        button:hover {
          background-color: #45a049;
        }
        @keyframes slideIn {
          from {
            transform: translateY(50px);
            opacity: 0;
          }
          to {
            transform: translateY(0);
            opacity: 1;
          }
        }
        @keyframes pulse {
          0% {
            transform: scale(1);
          }
          50% {
            transform: scale(1.1);
          }
          100% {
            transform: scale(1);
          }
        }
        @media (max-width: 768px) {
          .login-container {
            padding: 40px 20px;
          }
          h1 {
            font-size: 28px;
          }
          input {
            width: 90%;
          }
          button {
            width: 95%;
          }
        }
      </style>
    </head>
    <body>
      <div id="particles-js"></div>
      <div class="login-container">
        <img src="/assets/logo.png" alt="App Logo" class="logo">
        <h1>Admin Login</h1>
        <form action="/login" method="POST">
          <input type="text" name="username" placeholder="Username" required>
          <input type="password" name="password" placeholder="Password" required>
          <button type="submit">Login</button>
        </form>
      </div>
      <script>
        particlesJS("particles-js", {
          particles: {
            number: { value: 150, density: { enable: true, value_area: 800 } },
            color: { value: "#ffffff" },
            shape: {
              type: "circle",
              stroke: { width: 0, color: "#000000" },
              polygon: { nb_sides: 5 }
            },
            opacity: {
              value: 0.5,
              random: false,
              anim: { enable: false }
            },
            size: {
              value: 5,
              random: true,
              anim: { enable: false }
            },
            line_linked: {
              enable: true,
              distance: 150,
              color: "#ffffff",
              opacity: 0.4,
              width: 1
            },
            move: {
              enable: true,
              speed: 4,
              direction: "none",
              random: false,
              straight: false,
              out_mode: "out",
              bounce: false,
              attract: { enable: false }
            }
          },
          interactivity: {
            detect_on: "canvas",
            events: {
              onhover: { enable: true, mode: "repulse" },
              onclick: { enable: true, mode: "push" },
              resize: true
            },
            modes: {
              grab: { distance: 400, line_linked: { opacity: 1 } },
              bubble: { distance: 400, size: 40, duration: 2, opacity: 8, speed: 3 },
              repulse: { distance: 200, duration: 0.4 },
              push: { particles_nb: 4 },
              remove: { particles_nb: 2 }
            }
          },
          retina_detect: true
        });
      </script>
    </body>
    </html>
  `);
});

// Handle Login
app.post('/login', (req, res) => {
  const { username, password } = req.body;
  if (
    username === process.env.ADMIN_USERNAME &&
    password === process.env.ADMIN_PASSWORD
  ) {
    // Save login data to localStorage (in the browser context)
    res.send(`
      <script>
        localStorage.setItem('loggedIn', 'true');
        localStorage.setItem('username', '${username}');
        window.location.href = '/home';
      </script>
    `);
  } else {
    res.send(`
      <script>
        alert('Invalid Username or Password');
        window.location.href = '/'; // Redirect back to login page
      </script>
    `);
  }
});


// MongoDB URI from .env
const mongoURI = process.env.MONGO_URI;
const client = new MongoClient(mongoURI, { useNewUrlParser: true, useUnifiedTopology: true });

async function fetchUsers() {
  try {
    await client.connect();
    const db = client.db('Zaibten');
    const usersCollection = db.collection('users');
    const users = await usersCollection.find().toArray();
    return users;
  } catch (err) {
    console.error('Error fetching users:', err);
    return [];
  }
}

app.get('/home', async (req, res) => {
  const users = await fetchUsers();
  
  // Example dynamic data based on fetched users
  const totalUsers = users.length;
  const revenue = totalUsers * 100; // Example calculation
  const activeSessions = Math.floor(totalUsers * 0.3); // Example: 30% of users
  const positiveFeedback = 80; // Example static feedback percentage

  // Chart data (dynamic examples)
  const salesData = [30, 40, 50, 60, totalUsers];
  const engagementData = [120, 200, 150, totalUsers];
  const growthData = [totalUsers * 0.6, totalUsers * 0.4];
  const revenueSources = [totalUsers * 0.75, totalUsers * 0.25];

  // Render the HTML with the fetched users
  let userHTML = users.map(user => `
    <tr>
      <td>${user.name}</td>
      <td>${user.email}</td>
      <td>${user.password}</td>
    </tr>
  `).join('');

  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Admin Dashboard</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <style>
        body {
          font-family: 'Poppins', sans-serif;
          margin: 0;
          background-color: #f1f3f5;
        }

        .sidebar {
          height: 100vh;
          width: 250px;
          position: fixed;
          left: 0;
          top: 0;
          background: #1c1e26;
          color: #fff;
          padding: 20px;
          transition: 0.3s;
        }

        .sidebar.collapsed {
          width: 80px;
        }

        .sidebar .logo {
          font-size: 24px;
          text-align: center;
          margin-bottom: 20px;
          font-weight: bold;
        }

        .sidebar ul {
          padding: 0;
          list-style-type: none;
        }

        .sidebar ul li {
          padding: 15px;
          text-align: left;
          cursor: pointer;
          color: #b3b3b3;
          transition: 0.3s;
        }

        .sidebar a {
          text-decoration: none;
          color: #b3b3b3;
          transition: 0.3s;
        }

        .sidebar ul li:hover {
          background-color: #2c2f38;
          color: white;
        }

        .content {
          margin-left: 250px;
          padding: 20px;
          transition: 0.3s;
        }

        .content.collapsed {
          margin-left: 80px;
        }

        .navbar {
          background: #ffffff;
          color: #495057;
          padding: 15px;
          border-bottom: 1px solid #dee2e6;
        }

        .card {
          background-color: #fff;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          border: none;
          border-radius: 10px;
          padding: 20px;
          color: #495057;
          transition: transform 0.3s ease-in-out;
        }

        .card:hover {
          transform: translateY(-5px);
          background-color: #e9ecef;
        }

        .table-container {
          margin-top: 30px;
          overflow-x: auto;
          border-radius: 10px;
        }

        table {
          width: 100%;
          border-collapse: collapse;
          border-radius: 10px;
          overflow: hidden;
        }

        th, td {
          padding: 12px;
          text-align: left;
          border-bottom: 1px solid #ddd;
        }

        th {
          background-color: #2c3e50;
          color: black;
        }

        tr:nth-child(even) {
          background-color: #f2f2f2;
        }

        tr:hover {
          background-color: #f1f1f1;
        }

        .table-container {
          max-height: 400px;
          overflow-y: auto;
          display: block;
        }

        .table-container table {
          width: 100%;
          table-layout: fixed;
        }

        .table-container table th, .table-container table td {
          word-wrap: break-word;
        }

        /* Custom Edit and Delete Icons */
        .edit-icon,
        .delete-icon {
          font-size: 20px;
          padding: 8px;
          cursor: pointer;
          border-radius: 4px;
          margin-right: 10px;
          transition: background-color 0.3s;
        }

        .edit-icon {
          color: #4CAF50;
        }

        .edit-icon:hover {
          background-color: #e8f5e9;
        }

        .delete-icon {
          color: #f44336;
        }

        .delete-icon:hover {
          background-color: #ffebee;
        }

        @media (max-width: 768px) {
          .sidebar {
            width: 80px;
          }

          .content {
            margin-left: 80px;
          }
        }
      </style>
    </head>
    <body>
      <div class="sidebar">
        <div class="logo">Admin Panel</div>
        <ul>
          <li>Dashboard</li>
          <li><a href="/logout">Logout</a></li>
        </ul>
      </div>
      <div class="content">
        <div class="navbar">Welcome to the Admin Dashboard</div>
        <div class="container mt-4">
        <div class="row">
            <div class="col-md-3">
              <div class="card">
                <h5>Total Users</h5>
                <p>${totalUsers}</p>
              </div>
            </div>
            <div class="col-md-3">
              <div class="card">
                <h5>Revenue</h5>
                <p>$${revenue}</p>
              </div>
            </div>
            <div class="col-md-3">
              <div class="card">
                <h5>Active Sessions</h5>
                <p>${activeSessions}</p>
              </div>
            </div>
            <div class="col-md-3">
              <div class="card">
                <h5>Feedback</h5>
                <p>${positiveFeedback}% Positive</p>
              </div>
            </div>
          </div>
<hr>
          <br>
          <div class="row">
            <div class="col-12">
              <h4>Users Information</h4>
              <div class="table-container">
                <table class="table table-bordered table-striped">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Email</th>
                      <th>Password</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${userHTML}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <hr>
          <br>

          <div class="row canvas-container">
            <div class="col-md-6">
              <div class="card">
                <h5>Sales Analytics</h5>
                <canvas id="salesChart"></canvas>
              </div>
            </div>
            <div class="col-md-6">
              <div class="card">
                <h5>User Engagement</h5>
                <canvas id="engagementChart"></canvas>
              </div>
            </div>
            <div class="col-md-6">
              <div class="card">
                <h5>Growth Rate</h5>
                <canvas id="growthChart"></canvas>
              </div>
            </div>
            <div class="col-md-6">
              <div class="card">
                <h5>Revenue Trends</h5>
                <canvas id="revenueChart"></canvas>
              </div>
            </div>
          </div>
        </div>
      </div>

      <script>
         // Check if the user is logged in
        if (!localStorage.getItem('loggedIn')) {
          window.location.href = '/';
        }
        // Sales Chart
        const salesChartCtx = document.getElementById('salesChart').getContext('2d');
        new Chart(salesChartCtx, {
          type: 'line',
          data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
            datasets: [{
              label: 'Sales',
              data: ${JSON.stringify(salesData)},
              borderColor: 'rgba(173, 216, 230, 1)',
              backgroundColor: 'rgba(173, 216, 230, 0.2)',
              fill: true,
              tension: 0.4,
            }]
          }
        });

        // Engagement Chart
        const engagementChartCtx = document.getElementById('engagementChart').getContext('2d');
        new Chart(engagementChartCtx, {
          type: 'bar',
          data: {
            labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
            datasets: [{
              label: 'Engagement',
              data: ${JSON.stringify(engagementData)},
              backgroundColor: 'rgba(144, 238, 144, 0.6)',
              borderColor: 'rgba(144, 238, 144, 1)',
            }]
          }
        });

        // Growth Rate Chart
        const growthChartCtx = document.getElementById('growthChart').getContext('2d');
        new Chart(growthChartCtx, {
          type: 'pie',
          data: {
            labels: ['Growth 2023', 'Growth 2024'],
            datasets: [{
              label: 'Growth Rate',
              data: ${JSON.stringify(growthData)},
              backgroundColor: ['#ffb6c1', '#add8e6'],
            }]
          }
        });

        // Revenue Trends Chart
        const revenueChartCtx = document.getElementById('revenueChart').getContext('2d');
        new Chart(revenueChartCtx, {
          type: 'doughnut',
          data: {
            labels: ['Revenue from Sales', 'Revenue from Ads'],
            datasets: [{
              label: 'Revenue Sources',
              data: ${JSON.stringify(revenueSources)},
              backgroundColor: ['#f0e68c', '#dda0dd'],
            }]
          }
        });
      </script>
    </body>
    </html>
  `);
});




// Logout Route
app.get('/logout', (req, res) => {
  res.send(`
    <script>
      if (confirm('Are you sure you want to log out?')) {
        localStorage.removeItem('loggedIn');
        localStorage.removeItem('username');
        window.location.href = '/';
      } else {
        window.location.href = '/home';  // Redirect back to the home page if canceled
      }
    </script>
  `);
});


// Start Server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
