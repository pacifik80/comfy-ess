import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "ess_selector_widget",
  async getCustomWidgets() {
    return {
      "ess_selector_widget": (node, inputName, inputData) => {
        const container = document.createElement('div');
        let scenes = inputData.value || [];

        const renderScenes = () => {
          container.innerHTML = ''; 

          scenes.forEach((scene, index) => {
            const sceneLabel = document.createElement('label');
            sceneLabel.textContent = `Scene ${index + 1}`;

            const sceneInput = document.createElement('input');
            sceneInput.type = 'text';
            sceneInput.value = scene[0] || '';
            sceneInput.placeholder = 'Scene reference';
            sceneInput.onchange = (e) => {
              scenes[index][0] = e.target.value;
            };

            const weightInput = document.createElement('input');
            weightInput.type = 'number';
            weightInput.value = scene[1];
            weightInput.step = 0.1;
            weightInput.style.marginLeft = '5px';
            weightInput.onchange = (e) => {
              scenes[index][1] = parseFloat(e.target.value);
            };

            container.appendChild(sceneLabel);
            container.appendChild(sceneInput);
            container.appendChild(weightInput);
            container.appendChild(document.createElement('br'));
          });

          const addButton = document.createElement('button');
          addButton.textContent = '+ Add Scene';
          addButton.onclick = () => {
            scenes.push(['', 1.0]);
            renderScenes();
          };

          const removeButton = document.createElement('button');
          removeButton.textContent = '- Remove Scene';
          removeButton.onclick = () => {
            if (scenes.length > 1) {
              scenes.pop();
              renderScenes();
            }
          };

          container.appendChild(addButton);
          container.appendChild(removeButton);
        };

        renderScenes();

        return {
          element: container,
          getValue: () => scenes
        };
      }
    };
  }
});
