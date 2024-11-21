
// Draw a line between two shapes
function drawLine(start, end) {
      const startX = parseFloat(start.getAttribute("data-x")) + 50; // Center of shape
      const startY = parseFloat(start.getAttribute("data-y")) + 25;
      const endX = parseFloat(end.getAttribute("data-x")) + 50;
      const endY = parseFloat(end.getAttribute("data-y")) + 25;

      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", startX);
      line.setAttribute("y1", startY);
      line.setAttribute("x2", endX);
      line.setAttribute("y2", endY);
      line.classList.add("line");

      svg = document.getElementById("lines");
      svg.appendChild(line);
}

document.addEventListener("DOMContentLoaded", () => {
    const toolbar = document.getElementById("toolbar");
    const canvas = document.getElementById("canvas");
    const svg = document.getElementById("lines");

    let draggedElement = null;
    let shapes = [];
    let isConnecting = false;
    let startShape = null;

    // Create and drag shapes
    toolbar.addEventListener("dragstart", (e) => {
      if (e.target.classList.contains("draggable")) {
        draggedElement = e.target.cloneNode(true);
        draggedElement.classList.add("placed");
      }
    });

    canvas.addEventListener("dragover", (e) => e.preventDefault());

    canvas.addEventListener("drop", (e) => {
      if (draggedElement) {
        const canvasRect = canvas.getBoundingClientRect();
        draggedElement.style.position = "absolute";
        draggedElement.style.left = `${e.clientX - canvasRect.left - 50}px`;
        draggedElement.style.top = `${e.clientY - canvasRect.top - 25}px`;
        draggedElement.setAttribute("data-x", e.clientX - canvasRect.left);
        draggedElement.setAttribute("data-y", e.clientY - canvasRect.top);
        draggedElement.addEventListener("mousedown", startConnection);
        shapes.push(draggedElement);
        canvas.appendChild(draggedElement);
      }
      draggedElement = null;
    });

    // Start connecting shapes
    function startConnection(e) {
      if (!isConnecting) {
        isConnecting = true;
        startShape = e.target;
      } else {
        isConnecting = false;
        const endShape = e.target;
        if (startShape !== endShape) {
          drawLine(startShape, endShape);
        }
      }
    }

    // Drag to reposition shapes
    canvas.addEventListener("mousedown", (e) => {
      if (e.target.classList.contains("placed")) {
        const shape = e.target;
        const onMove = (event) => {
          const canvasRect = canvas.getBoundingClientRect();
          const x = event.clientX - canvasRect.left - 50;
          const y = event.clientY - canvasRect.top - 25;

          shape.style.left = `${x}px`;
          shape.style.top = `${y}px`;
          shape.setAttribute("data-x", x);
          shape.setAttribute("data-y", y);

          updateLines();
        };

        const onStop = () => {
          canvas.removeEventListener("mousemove", onMove);
          canvas.removeEventListener("mouseup", onStop);
        };

        canvas.addEventListener("mousemove", onMove);
        canvas.addEventListener("mouseup", onStop);
      }
    });

    // Update line positions after moving shapes
    function updateLines() {
      const lines = svg.querySelectorAll(".line");
      lines.forEach((line) => {
        const startShape = shapes.find((s) => s.getAttribute("data-x") == line.getAttribute("x1") - 50);
        const endShape = shapes.find((s) => s.getAttribute("data-x") == line.getAttribute("x2") - 50);
        if (startShape && endShape) {
          line.setAttribute("x1", parseFloat(startShape.getAttribute("data-x")) + 50);
          line.setAttribute("y1", parseFloat(startShape.getAttribute("data-y")) + 25);
          line.setAttribute("x2", parseFloat(endShape.getAttribute("data-x")) + 50);
          line.setAttribute("y2", parseFloat(endShape.getAttribute("data-y")) + 25);
        }
      });
    }


});


function format_workflow() {
                shapeList = document.querySelectorAll(".div_workflow_shape");
                if (shapeList == null) {
                    return;
                }
                
                canvas = document.getElementById("canvas");
                canvas_width = canvas.offsetWidth;
                canvas_height = canvas.offsetHeight;

                coordinate_x = 0
                coordinate_y = 0
                gap_height = 20.0

                shapeList.forEach(function(element) {
                    
                    elem_width = element.offsetWidth;
                    elem_height = element.offsetHeight;

                    coordinate_x = (canvas_width - elem_width)/2.0 
                    coordinate_y += (elem_height + gap_height)

                    element.style.left = `${coordinate_x}px`;
                    element.style.top = `${coordinate_y}px`;
                    element.style.position = "absolute";

                    element.setAttribute("data-x", coordinate_x);
                    element.setAttribute("data-y", coordinate_y);

                });

                // draw lines
                var prevElem = null;
                shapeList.forEach(function(element) {
                    if (prevElem == null) {
                        prevElem = element;
                    } else {
                        drawLine(prevElem, element);
                        prevElem = element;
                    }
                });

};


/** 
* format workflow group by 
  agent_name
  process_id
  timestamp
*/
function format_workflow_group() {

    canvas = document.getElementById("canvas");
    canvas_width = canvas.offsetWidth;
    canvas_height = canvas.offsetHeight;

    divAgent = document.querySelectorAll(".div_workflow_agent");
    if (divAgent == null) {
        return;
    }

    num_agent = divAgent.length;
    section_width = (canvas_width/num_agent);


    // grid layout parameters
    grid_x_max_cnt = 3
    grid_x_col_gap = 20.0

    div_agent_margin_top = 30.0;

    var agent_section_id = 0;
    divAgent.forEach(function(elementAgent) {

        var agentName = elementAgent.getAttribute("agent_name")
        var agentWorkflowList = elementAgent.querySelectorAll(".div_workflow_shape");

        coordinateX = 0;
        coordinateY = 0;

        // process_id
        var prevProcessId = null;

        //start 
        var curProcessGridCnt = 0;

        // start grid coordinate
        var processGridX = section_width/2.0;
        var processGridY = 0.0;
        var processGridWidth = 0;
        var processGridHeight = 0;
        var processGridCardMaxX = 100.0;
        var processGridCardMaxY = 50.0;
        var processGridCardRowGap = 10.0;
        var processGridCardColGap = 10.0;

        var processGridGroupRowGap = 80;

        // start point of group position
        processGridX = agent_section_id * section_width + section_width/2.0;
        processGridY = div_agent_margin_top;

        agentWorkflowList.forEach(function(element) {
            
            elemWidth = element.offsetWidth;
            elemHeight = element.offsetHeight;

            // process_id
            var elementProcessId = element.getAttribute("process_id");
            if (elementProcessId == null) {
              // new process id Todo random
              elementProcessId = "";
            }

            var ifNewProcessGroup = false;
            if (prevProcessId == null) {
                prevProcessId = elementProcessId;
                ifNewProcessGroup = true;
            } else {
                ifNewProcessGroup = !(elementProcessId == prevProcessId)

            }
            if (ifNewProcessGroup) {
                // x: reset back to start
                // y: change to new
                // move group start coordinate
                processGridX = agent_section_id * section_width + section_width/2.0;                
                processGridY += processGridGroupRowGap;

                //update previous id
                prevProcessId = elementProcessId;
                curProcessGridCnt = 0;
            } else {

            }

            // cur Process Add new Cnt
            curProcessGridCnt += 1;
            // cur div add width and height
            curProcessGridCol = (curProcessGridCnt >= grid_x_max_cnt)? grid_x_max_cnt: curProcessGridCnt;
            curProcessGridRow = Math.ceil(curProcessGridCnt/grid_x_max_cnt);
            // calculate cur div group relative grid card cnt (x_cnt, y_cnt)

            processGridWidth = (curProcessGridCol - 1) * ( Math.max(processGridCardMaxX, elemWidth) + processGridCardColGap);
            processGridHeight = (curProcessGridRow - 1) * ( Math.max(processGridCardMaxY, elemHeight) + processGridCardRowGap);
            // gorup start - > relative postion

            // calculate new elem coordinate
            elemWidth = element.offsetWidth;
            elemHeight = element.offsetHeight;

            coordinateX = (processGridX + processGridWidth);
            coordinateY = (processGridY + processGridHeight );


            element.style.left = `${coordinateX}px`;
            element.style.top = `${coordinateY}px`;
            element.style.position = "absolute";
            element.setAttribute("data-x", coordinateX);
            element.setAttribute("data-y", coordinateY);


        });

        // move agent offset to new agent
        agent_section_id += 1;
    });

}

