function plot_values(filename) {

    filename = "./data/" + filename;
    svg.selectAll("*").remove();

    color = ["FF6A5C","056571"];
    // Load data

    d3.csv(filename, function(error, data) {
        data.forEach(function(d) {
            d.x = +d.x;
            d.y = +d.y;
            d.type = +d.type;
        });

        var xValueR = function(d) { return d.x;};
        var yValueR = function(d) { return d.y;};

        var labels = ['Random', 'Adaptive']
        xScale.domain([d3.min(data, xValueR), d3.max(data, xValueR)]);
        yScale.domain([d3.min(data, yValueR), d3.max(data, yValueR)]);


        svg.append("g")
          .attr("class", "axis")
          .attr("transform", "translate(0, "+(h-pad)+")")
          .call(xAxis);

        svg.append("g")
          .attr("class", "axis")
          .attr("transform", "translate("+(left_pad-pad)+", 0)")
          .call(yAxis);

        svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", left_pad-120)
        .attr("x",h-500)
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("Component 1");

        svg.append("text")
        //.attr("transform", "rotate(-20)")
        .attr("y", left_pad+200)
        .attr("x",h+500)
        .attr("dy", "3em")
        .style("text-anchor", "middle")
        .text("Component 2");


        svg.selectAll("circle")
            .data(data)
            .enter()
            .append("circle")
            .attr("r", 3)
            .attr("cx", function(d){
                return xScale(d.x);
            })
            .attr("cy", function(d){
                return yScale(d.y);
            })
            .style("fill", function(d) {
                return color[d.type-1];
            })
            .attr("stroke", "black")
            ;


        var legend = svg.append("g")
      .attr("class", "legend")
      .attr("x", w - 85)
      .attr("y", 35)
      .attr("height", 100)
      .attr("width", 100);

    legend.selectAll('g').data(color)
      .enter()
      .append('g')
      .each(function(d, i) {
        var g = d3.select(this);
        g.append("rect")
          .attr("x", w - 100)
          .attr("y", i*25 + 20)
          .attr("width", 10)
          .attr("height", 10)
          .style("fill", color[i]);

        g.append("text")
          .attr("x", w - 85)
          .attr("y", i * 25 + 30)
          .attr("height",30)
          .attr("width",100)
          .text(labels[i]);

      })
});


}
